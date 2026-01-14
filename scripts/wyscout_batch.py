#!/usr/bin/env python3
"""
Wyscout Batch Downloader - Interactive URL Capture Mode

This script helps you batch download multiple Wyscout videos by:
1. Opening a browser where you manually navigate Wyscout
2. Automatically capturing download URLs when you click "Start Download"
3. Downloading all captured videos after you're done

This approach is more reliable since Wyscout's UI may change.

Usage:
    python wyscout_batch.py --interactive
    python wyscout_batch.py --urls urls.txt --download
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm


def download_video(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download video from signed URL with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name[:40]) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"[-] Failed to download {output_path.name}: {e}")
        return False


def extract_video_id(url: str) -> str:
    """Extract video ID from Wyscout CDN URL."""
    # URL pattern: https://cdn5download.wyscout.com/privateversions/5787516_691bb57c77ac0-hd.mp4?...
    path = urlparse(url).path
    filename = Path(path).stem  # e.g., "5787516_691bb57c77ac0-hd"
    return filename


def check_url_valid(url: str) -> bool:
    """Check if the signed URL is still valid (not expired)."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if 'Expires' in params:
        expires = int(params['Expires'][0])
        now = int(time.time())
        if now > expires:
            print(f"[!] URL expired at {datetime.fromtimestamp(expires)}")
            return False
        remaining = expires - now
        print(f"[+] URL valid for {remaining // 3600}h {(remaining % 3600) // 60}m")
    return True


def interactive_capture():
    """Open browser and capture download URLs as user navigates."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[-] Playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    captured_urls = []
    url_file = Path("captured_urls.txt")

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    WYSCOUT INTERACTIVE URL CAPTURE                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  1. A browser will open - log into Wyscout                                ║
║  2. Navigate to each match you want to download                           ║
║  3. Click: Wide Angle → Export → Download Video → Start Download          ║
║  4. The download URL will be automatically captured                       ║
║  5. Cancel the browser's download dialog (we'll download separately)      ║
║  6. Repeat for all matches                                                ║
║  7. Close browser when done - URLs saved to captured_urls.txt             ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    input("Press Enter to open browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        def on_request(request):
            url = request.url
            # Capture Wyscout CDN download URLs
            if "cdn" in url and "wyscout.com" in url and ".mp4" in url:
                if url not in captured_urls:
                    captured_urls.append(url)
                    video_id = extract_video_id(url)
                    print(f"\n[+] Captured URL #{len(captured_urls)}: {video_id}")
                    # Save immediately
                    with open(url_file, "a") as f:
                        f.write(url + "\n")

        page.on("request", on_request)

        # Navigate to Wyscout
        page.goto("https://platform.wyscout.com")

        print("\n[*] Browser opened. Navigate to matches and start downloads.")
        print("[*] URLs are saved to captured_urls.txt as they're captured.")
        print("[*] Close the browser when you're done.\n")

        # Wait for user to close browser
        try:
            page.wait_for_event("close", timeout=0)
        except:
            pass

        browser.close()

    print(f"\n[+] Captured {len(captured_urls)} URLs")
    print(f"[+] Saved to: {url_file.absolute()}")
    return captured_urls


def batch_download(urls: list, output_dir: Path, prefix: str = ""):
    """Download multiple videos from URLs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url or url.startswith("#"):
            continue

        video_id = extract_video_id(url)
        if prefix:
            filename = f"{prefix}_{i:02d}_{video_id}.mp4"
        else:
            filename = f"{video_id}.mp4"

        output_path = output_dir / filename

        if output_path.exists():
            print(f"[*] Skipping {filename} (already exists)")
            successful += 1
            continue

        print(f"\n[{i}/{len(urls)}] Downloading {filename}...")

        if not check_url_valid(url):
            print(f"[-] Skipping expired URL")
            failed += 1
            continue

        if download_video(url, output_path):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"[+] Downloaded: {successful}/{len(urls)} videos")
    if failed:
        print(f"[-] Failed: {failed} videos")


def main():
    parser = argparse.ArgumentParser(
        description="Wyscout video batch downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode - capture URLs while browsing
    python wyscout_batch.py --interactive

    # Download from previously captured URLs
    python wyscout_batch.py --urls captured_urls.txt --download

    # Download single URL
    python wyscout_batch.py --url "https://cdn5download..." -o match.mp4

    # Check if URL is still valid
    python wyscout_batch.py --check "https://cdn5download..."
        """,
    )

    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode - open browser to capture URLs")
    parser.add_argument("--urls", type=Path,
                        help="File containing URLs (one per line)")
    parser.add_argument("--url", help="Single URL to download")
    parser.add_argument("--download", "-d", action="store_true",
                        help="Download videos from URL file")
    parser.add_argument("--output", "-o", type=Path, default=Path("./downloads"),
                        help="Output directory or filename")
    parser.add_argument("--prefix", default="",
                        help="Prefix for downloaded filenames")
    parser.add_argument("--check", help="Check if a URL is still valid")

    args = parser.parse_args()

    if args.check:
        check_url_valid(args.check)
        return

    if args.interactive:
        urls = interactive_capture()
        if urls and input("\nDownload captured videos now? [y/N]: ").lower() == 'y':
            batch_download(urls, args.output, args.prefix)
        return

    if args.url:
        output_path = args.output if args.output.suffix == '.mp4' else args.output / f"{extract_video_id(args.url)}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if check_url_valid(args.url):
            download_video(args.url, output_path)
        return

    if args.urls and args.download:
        with open(args.urls) as f:
            urls = f.readlines()
        batch_download(urls, args.output, args.prefix)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
