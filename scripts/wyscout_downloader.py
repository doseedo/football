#!/usr/bin/env python3
"""
Wyscout Wide Angle Video Downloader

Automates downloading wide-angle match videos from Wyscout website.
Requires valid Wyscout credentials.

Usage:
    python wyscout_downloader.py --email your@email.com --password yourpass
    python wyscout_downloader.py --config credentials.json
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Page, Browser, TimeoutError as PlaywrightTimeout


@dataclass
class MatchInfo:
    """Information about a match to download."""
    country: str
    league: str
    home_team: str
    away_team: str
    output_filename: Optional[str] = None

    def get_filename(self) -> str:
        """Generate output filename if not specified."""
        if self.output_filename:
            return self.output_filename
        # Sanitize team names for filename
        home = re.sub(r'[^\w\s-]', '', self.home_team).strip().replace(' ', '_')
        away = re.sub(r'[^\w\s-]', '', self.away_team).strip().replace(' ', '_')
        return f"{home}_vs_{away}_wide_angle.mp4"


class WyscoutDownloader:
    """Automates Wyscout wide angle video downloads."""

    WYSCOUT_URL = "https://wyscout.com"
    LOGIN_URL = "https://wyscout.com/login"
    PLATFORM_URL = "https://platform.wyscout.com"

    def __init__(
        self,
        email: str,
        password: str,
        download_dir: str = "./downloads",
        headless: bool = False,
        timeout: int = 60000,
    ):
        self.email = email
        self.password = password
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Initialize browser and login."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            downloads_path=str(self.download_dir),
        )
        self.context = self.browser.new_context(
            accept_downloads=True,
            viewport={"width": 1920, "height": 1080},
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout)

    def close(self):
        """Clean up browser resources."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def login(self) -> bool:
        """Login to Wyscout platform."""
        print(f"[*] Logging in as {self.email}...")

        try:
            self.page.goto(self.LOGIN_URL)
            self.page.wait_for_load_state("networkidle")

            # Fill login form - adjust selectors based on actual page structure
            # These selectors may need updating based on Wyscout's current UI
            email_input = self.page.locator('input[type="email"], input[name="email"], #email')
            password_input = self.page.locator('input[type="password"], input[name="password"], #password')

            email_input.first.fill(self.email)
            password_input.first.fill(self.password)

            # Click login button
            login_button = self.page.locator('button[type="submit"], input[type="submit"], button:has-text("Login"), button:has-text("Sign in")')
            login_button.first.click()

            # Wait for redirect to platform
            self.page.wait_for_url(f"{self.PLATFORM_URL}/**", timeout=30000)
            print("[+] Login successful!")
            return True

        except PlaywrightTimeout:
            print("[-] Login failed - timeout waiting for platform redirect")
            return False
        except Exception as e:
            print(f"[-] Login failed: {e}")
            return False

    def navigate_to_match(self, match: MatchInfo) -> bool:
        """Navigate to a specific match in the Wyscout platform."""
        print(f"[*] Navigating to {match.home_team} vs {match.away_team}...")

        try:
            # Go to matches/videos section
            # Navigate through: Country -> League -> Match
            # This is a simplified flow - actual navigation may require more steps

            # Look for competition/country selector
            self.page.wait_for_load_state("networkidle")

            # Search for the match directly if search is available
            search_box = self.page.locator('input[type="search"], input[placeholder*="Search"], .search-input')
            if search_box.count() > 0:
                search_query = f"{match.home_team} {match.away_team}"
                search_box.first.fill(search_query)
                self.page.keyboard.press("Enter")
                self.page.wait_for_load_state("networkidle")
                time.sleep(2)

            # Alternative: Navigate through menus
            # Click on country
            country_elem = self.page.locator(f'text="{match.country}"').first
            if country_elem.is_visible():
                country_elem.click()
                self.page.wait_for_load_state("networkidle")

            # Click on league
            league_elem = self.page.locator(f'text="{match.league}"').first
            if league_elem.is_visible():
                league_elem.click()
                self.page.wait_for_load_state("networkidle")

            # Find and click on the match
            # Look for match card/row containing both team names
            match_selector = f'//*[contains(text(), "{match.home_team}") and contains(text(), "{match.away_team}")]'
            match_elem = self.page.locator(match_selector).first

            if not match_elem.is_visible():
                # Try partial match
                match_elem = self.page.locator(f'text=/{match.home_team}.*{match.away_team}/i').first

            match_elem.click()
            self.page.wait_for_load_state("networkidle")
            print("[+] Found match!")
            return True

        except Exception as e:
            print(f"[-] Failed to navigate to match: {e}")
            return False

    def select_wide_angle(self) -> bool:
        """Select the wide angle version of the video."""
        print("[*] Selecting wide angle version...")

        try:
            # Look for wide angle option - common patterns
            wide_angle_selectors = [
                'text="Wide angle"',
                'text="Wide Angle"',
                'text="wide angle"',
                '[data-video-type="wide"]',
                '.video-type-wide',
                'button:has-text("Wide")',
                '*:has-text("Wide angle version")',
            ]

            for selector in wide_angle_selectors:
                elem = self.page.locator(selector)
                if elem.count() > 0 and elem.first.is_visible():
                    elem.first.click()
                    self.page.wait_for_load_state("networkidle")
                    print("[+] Wide angle selected!")
                    return True

            # If no explicit wide angle button, it might be in a dropdown
            video_type_dropdown = self.page.locator('[class*="video-type"], [class*="version-select"], select')
            if video_type_dropdown.count() > 0:
                video_type_dropdown.first.click()
                time.sleep(1)
                wide_option = self.page.locator('text="Wide"').first
                wide_option.click()
                return True

            print("[!] Wide angle selector not found - video may already be wide angle")
            return True

        except Exception as e:
            print(f"[-] Failed to select wide angle: {e}")
            return False

    def download_video(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Click export and download the video."""
        print("[*] Initiating download...")

        try:
            # Click Export button
            export_selectors = [
                'button:has-text("Export")',
                'text="Export"',
                '[class*="export"]',
                'button:has-text("Download")',
            ]

            for selector in export_selectors:
                elem = self.page.locator(selector)
                if elem.count() > 0 and elem.first.is_visible():
                    elem.first.click()
                    break
            else:
                print("[-] Export button not found")
                return None

            self.page.wait_for_load_state("networkidle")
            time.sleep(2)

            # Click "Download video" in the export menu
            download_video_selectors = [
                'text="Download video"',
                'text="Download Video"',
                'button:has-text("Download video")',
                '[class*="download-video"]',
            ]

            for selector in download_video_selectors:
                elem = self.page.locator(selector)
                if elem.count() > 0 and elem.first.is_visible():
                    elem.first.click()
                    break

            time.sleep(2)

            # Start download - this should trigger the actual download
            # Wait for download to start
            with self.page.expect_download(timeout=120000) as download_info:
                # Click the final download button/link
                start_download_selectors = [
                    'text="Start download"',
                    'text="Start Download"',
                    'button:has-text("Start")',
                    'a[href*="cdn"][href*="download"]',
                    '[class*="start-download"]',
                ]

                for selector in start_download_selectors:
                    elem = self.page.locator(selector)
                    if elem.count() > 0 and elem.first.is_visible():
                        elem.first.click()
                        break

            download = download_info.value

            # Save to specified path or use suggested filename
            if output_path:
                final_path = output_path
            else:
                final_path = self.download_dir / download.suggested_filename

            print(f"[*] Downloading to {final_path}...")
            download.save_as(final_path)

            # Wait for download to complete
            print("[+] Download complete!")
            return final_path

        except PlaywrightTimeout:
            print("[-] Download timeout - the file may be too large or network is slow")
            return None
        except Exception as e:
            print(f"[-] Download failed: {e}")
            return None

    def capture_download_url(self) -> Optional[str]:
        """Capture the signed download URL instead of downloading directly."""
        print("[*] Capturing download URL...")

        captured_url = None

        def handle_request(request):
            nonlocal captured_url
            if "cdn" in request.url and "download" in request.url and ".mp4" in request.url:
                captured_url = request.url
                print(f"[+] Captured URL: {request.url[:100]}...")

        self.page.on("request", handle_request)

        try:
            # Go through export flow
            export_elem = self.page.locator('button:has-text("Export"), text="Export"').first
            export_elem.click()
            time.sleep(2)

            download_elem = self.page.locator('text="Download video"').first
            download_elem.click()
            time.sleep(2)

            start_elem = self.page.locator('text="Start download", a[href*="cdn"]').first
            start_elem.click()
            time.sleep(3)

            return captured_url

        except Exception as e:
            print(f"[-] Failed to capture URL: {e}")
            return None

    def download_match(self, match: MatchInfo) -> Optional[Path]:
        """Full workflow to download a match's wide angle video."""
        if not self.navigate_to_match(match):
            return None

        if not self.select_wide_angle():
            return None

        output_path = self.download_dir / match.get_filename()
        return self.download_video(output_path)


def download_from_url(url: str, output_path: str, chunk_size: int = 8192):
    """Download video directly from a captured URL."""
    import requests

    print(f"[*] Downloading from URL to {output_path}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"\r[*] Progress: {percent:.1f}%", end="", flush=True)

    print(f"\n[+] Downloaded {downloaded / (1024*1024):.1f} MB")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download wide angle videos from Wyscout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode with credentials
    python wyscout_downloader.py --email user@example.com --password mypass

    # Download specific match
    python wyscout_downloader.py --email user@example.com --password mypass \\
        --country "United States" --league "NCAA D1 Sun Belt" \\
        --home "Marshall Thundering Herd" --away "UCF Knights"

    # Download from captured URL (no browser needed)
    python wyscout_downloader.py --url "https://cdn5download..." --output match.mp4

    # Use config file for credentials
    python wyscout_downloader.py --config credentials.json
        """,
    )

    # Authentication
    parser.add_argument("--email", help="Wyscout email")
    parser.add_argument("--password", help="Wyscout password")
    parser.add_argument("--config", help="JSON config file with credentials")

    # Match selection
    parser.add_argument("--country", default="United States", help="Country name")
    parser.add_argument("--league", default="NCAA D1 Sun Belt", help="League/competition name")
    parser.add_argument("--home", help="Home team name")
    parser.add_argument("--away", help="Away team name")

    # Direct URL download
    parser.add_argument("--url", help="Direct download URL (skip browser automation)")

    # Output
    parser.add_argument("--output", "-o", help="Output filename")
    parser.add_argument("--download-dir", default="./downloads", help="Download directory")

    # Browser options
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--capture-url", action="store_true", help="Only capture URL, don't download")

    args = parser.parse_args()

    # Handle direct URL download
    if args.url:
        output = args.output or "wyscout_video.mp4"
        download_from_url(args.url, output)
        return

    # Load credentials
    email, password = args.email, args.password
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
            email = config.get("email", email)
            password = config.get("password", password)

    if not email or not password:
        parser.error("Email and password required (via --email/--password or --config)")

    # Create match info
    match = MatchInfo(
        country=args.country,
        league=args.league,
        home_team=args.home or "Marshall Thundering Herd",
        away_team=args.away or "UCF Knights",
        output_filename=args.output,
    )

    # Run downloader
    with WyscoutDownloader(
        email=email,
        password=password,
        download_dir=args.download_dir,
        headless=args.headless,
    ) as downloader:
        if not downloader.login():
            print("[-] Failed to login")
            return

        if args.capture_url:
            url = downloader.capture_download_url()
            if url:
                print(f"\n[+] Download URL:\n{url}")
        else:
            result = downloader.download_match(match)
            if result:
                print(f"\n[+] Video saved to: {result}")
            else:
                print("\n[-] Download failed")


if __name__ == "__main__":
    main()
