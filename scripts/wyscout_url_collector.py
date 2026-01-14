#!/usr/bin/env python3
"""
Wyscout URL Collector

Opens a browser with network interception to capture download URLs.
You navigate manually, script captures CDN URLs automatically.

Usage:
    python wyscout_url_collector.py
    python wyscout_url_collector.py --output my_urls.txt
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description="Capture Wyscout download URLs")
    parser.add_argument("--output", "-o", default="wyscout_urls.txt", help="Output file for URLs")
    parser.add_argument("--start-url", default="https://wyscout.hudl.com", help="Starting URL")
    args = parser.parse_args()

    captured_urls = []
    output_file = Path(args.output)

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     WYSCOUT URL COLLECTOR                                  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  1. Browser will open - LOG IN to Wyscout                                 ║
║  2. Navigate to each match you want                                       ║
║  3. Click: Wide Angle → Export → Download Video → Start Download          ║
║  4. URLs are captured automatically (watch this terminal)                 ║
║  5. Close browser when done                                               ║
║                                                                           ║
║  URLs saved to: {:<55} ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """.format(str(output_file)))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Intercept ALL requests at the browser level
        def handle_request(request):
            url = request.url
            if "cdn" in url and "wyscout" in url and ".mp4" in url:
                if url not in captured_urls:
                    captured_urls.append(url)
                    # Extract video ID for display
                    match = re.search(r'/(\d+_[a-f0-9]+-hd\.mp4)', url)
                    video_id = match.group(1) if match else url[:60]
                    print(f"\n✓ Captured #{len(captured_urls)}: {video_id}")

                    # Save immediately
                    with open(output_file, "a") as f:
                        f.write(url + "\n")

        def handle_response(response):
            url = response.url
            # Also check responses for redirects to video URLs
            if "cdn" in url and ".mp4" in url:
                if url not in captured_urls:
                    captured_urls.append(url)
                    print(f"\n✓ Captured #{len(captured_urls)} (response): {url[:70]}...")
                    with open(output_file, "a") as f:
                        f.write(url + "\n")

        # Listen to all requests and responses
        page.on("request", handle_request)
        page.on("response", handle_response)

        # Also intercept at context level for new tabs/popups
        context.on("request", handle_request)
        context.on("response", handle_response)

        # Handle new pages (popups/new tabs)
        def handle_new_page(new_page):
            new_page.on("request", handle_request)
            new_page.on("response", handle_response)
            # Check if the new page URL itself is a video
            if "cdn" in new_page.url and ".mp4" in new_page.url:
                if new_page.url not in captured_urls:
                    captured_urls.append(new_page.url)
                    print(f"\n✓ Captured #{len(captured_urls)} (new tab): {new_page.url[:70]}...")
                    with open(output_file, "a") as f:
                        f.write(new_page.url + "\n")

        context.on("page", handle_new_page)

        # Navigate to Wyscout
        page.goto(args.start_url)

        print("\n[*] Browser opened. Navigate to matches and start downloads.")
        print("[*] URLs are saved to file as they're captured.")
        print("[*] Close the browser window when done.\n")

        # Wait for browser to close
        try:
            page.wait_for_event("close", timeout=0)
        except:
            pass

        # Wait a bit for any pending captures
        try:
            context.close()
        except:
            pass

        browser.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"Captured {len(captured_urls)} URLs")
    print(f"Saved to: {output_file.absolute()}")

    if captured_urls:
        print(f"\nTo download all videos, run:")
        print(f"  python3 scripts/wyscout_batch.py --urls {output_file} --download")


if __name__ == "__main__":
    main()
