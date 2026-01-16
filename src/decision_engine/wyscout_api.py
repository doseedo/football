"""
Wyscout API Client

Fetches event data from Wyscout API for use with the decision analysis system.

Usage:
    from wyscout_api import WyscoutAPI

    api = WyscoutAPI(api_key="your_key")
    events = api.get_match_events(match_id=12345)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class WyscoutConfig:
    """Wyscout API configuration."""
    api_key: str
    base_url: str = "https://apirest.wyscout.com/v3"
    # Alternative base URLs if needed
    # base_url: str = "https://api.wyscout.com/v2"


class WyscoutAPI:
    """
    Client for Wyscout REST API.

    Wyscout API docs: https://apidocs.wyscout.com/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        base_url: str = "https://apirest.wyscout.com/v3"
    ):
        """
        Initialize Wyscout API client.

        Args:
            api_key: Wyscout API key (or set WYSCOUT_API_KEY env var)
            config_path: Path to JSON config with api_key
            base_url: API base URL
        """
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")

        # Load API key from various sources
        self.api_key = api_key or os.environ.get('WYSCOUT_API_KEY')

        if config_path and not self.api_key:
            with open(config_path) as f:
                config = json.load(f)
                self.api_key = config.get('api_key', config.get('apiKey'))

        if not self.api_key:
            raise ValueError(
                "Wyscout API key required. Provide via:\n"
                "  - api_key parameter\n"
                "  - WYSCOUT_API_KEY environment variable\n"
                "  - config_path JSON file with 'api_key' field"
            )

        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        })

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.5  # seconds between requests

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(url, params=params)
            self._last_request_time = time.time()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise ValueError("Invalid API key or unauthorized access")
            elif response.status_code == 403:
                raise ValueError("Access forbidden - check your API subscription")
            elif response.status_code == 404:
                raise ValueError(f"Resource not found: {endpoint}")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded - wait and retry")
            else:
                raise ValueError(f"API error {response.status_code}: {response.text}")

    # =========================================================================
    # Match Events
    # =========================================================================

    def get_match_events(self, match_id: int) -> List[Dict]:
        """
        Get all events for a match.

        Args:
            match_id: Wyscout match ID

        Returns:
            List of event dictionaries
        """
        data = self._request(f"matches/{match_id}/events")
        events = data.get('events', data) if isinstance(data, dict) else data
        print(f"Fetched {len(events)} events for match {match_id}")
        return events

    def get_match_info(self, match_id: int) -> Dict:
        """Get match metadata."""
        return self._request(f"matches/{match_id}")

    def get_match_formations(self, match_id: int) -> Dict:
        """Get team formations for a match."""
        return self._request(f"matches/{match_id}/formations")

    # =========================================================================
    # Competitions & Seasons
    # =========================================================================

    def get_competitions(self) -> List[Dict]:
        """Get list of available competitions."""
        data = self._request("competitions")
        return data.get('competitions', data)

    def get_seasons(self, competition_id: int) -> List[Dict]:
        """Get seasons for a competition."""
        data = self._request(f"competitions/{competition_id}/seasons")
        return data.get('seasons', data)

    def get_matches(self, season_id: int) -> List[Dict]:
        """Get all matches for a season."""
        data = self._request(f"seasons/{season_id}/matches")
        return data.get('matches', data)

    # =========================================================================
    # Teams & Players
    # =========================================================================

    def get_team(self, team_id: int) -> Dict:
        """Get team information."""
        return self._request(f"teams/{team_id}")

    def get_team_squad(self, team_id: int, season_id: int) -> List[Dict]:
        """Get team squad for a season."""
        data = self._request(f"teams/{team_id}/squad", params={'seasonId': season_id})
        return data.get('squad', data)

    def get_player(self, player_id: int) -> Dict:
        """Get player information."""
        return self._request(f"players/{player_id}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def save_match_events(self, match_id: int, output_path: str) -> str:
        """
        Fetch and save match events to JSON file.

        Args:
            match_id: Wyscout match ID
            output_path: Path to save JSON file

        Returns:
            Path to saved file
        """
        events = self.get_match_events(match_id)

        # Also get match info for context
        try:
            match_info = self.get_match_info(match_id)
        except:
            match_info = {}

        data = {
            'match_id': match_id,
            'match_info': match_info,
            'events': events,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(events)} events to {output_path}")
        return str(output_path)

    def fetch_season_events(
        self,
        season_id: int,
        output_dir: str,
        max_matches: Optional[int] = None
    ) -> List[str]:
        """
        Fetch events for all matches in a season.

        Args:
            season_id: Wyscout season ID
            output_dir: Directory to save JSON files
            max_matches: Maximum matches to fetch (None for all)

        Returns:
            List of paths to saved files
        """
        matches = self.get_matches(season_id)

        if max_matches:
            matches = matches[:max_matches]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for i, match in enumerate(matches):
            match_id = match.get('matchId', match.get('id'))
            output_path = output_dir / f"{match_id}.json"

            print(f"[{i+1}/{len(matches)}] Fetching match {match_id}...")

            try:
                self.save_match_events(match_id, str(output_path))
                saved_files.append(str(output_path))
            except Exception as e:
                print(f"  Error: {e}")
                continue

        return saved_files


def test_connection(api_key: str) -> bool:
    """Test API connection with provided key."""
    try:
        api = WyscoutAPI(api_key=api_key)
        competitions = api.get_competitions()
        print(f"Connection successful! Found {len(competitions)} competitions")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Wyscout API Client')
    parser.add_argument('--api-key', '-k', help='Wyscout API key (or set WYSCOUT_API_KEY env)')
    parser.add_argument('--config', '-c', help='Path to config JSON with api_key')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Test connection
    test_parser = subparsers.add_parser('test', help='Test API connection')

    # Get match events
    events_parser = subparsers.add_parser('events', help='Get match events')
    events_parser.add_argument('match_id', type=int, help='Wyscout match ID')
    events_parser.add_argument('--output', '-o', help='Output JSON path')

    # List competitions
    comp_parser = subparsers.add_parser('competitions', help='List competitions')

    # List matches
    matches_parser = subparsers.add_parser('matches', help='List matches for a season')
    matches_parser.add_argument('season_id', type=int, help='Season ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)

    # Initialize API
    try:
        api = WyscoutAPI(
            api_key=args.api_key,
            config_path=args.config
        )
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    if args.command == 'test':
        api.get_competitions()
        print("API connection successful!")

    elif args.command == 'events':
        events = api.get_match_events(args.match_id)
        if args.output:
            api.save_match_events(args.match_id, args.output)
        else:
            print(json.dumps(events[:5], indent=2))
            print(f"... ({len(events)} total events)")

    elif args.command == 'competitions':
        comps = api.get_competitions()
        for comp in comps[:20]:
            print(f"  {comp.get('wyId', comp.get('id'))}: {comp.get('name')}")
        if len(comps) > 20:
            print(f"  ... ({len(comps)} total)")

    elif args.command == 'matches':
        matches = api.get_matches(args.season_id)
        for match in matches[:20]:
            home = match.get('home', {}).get('name', 'Unknown')
            away = match.get('away', {}).get('name', 'Unknown')
            match_id = match.get('matchId', match.get('id'))
            print(f"  {match_id}: {home} vs {away}")
        if len(matches) > 20:
            print(f"  ... ({len(matches)} total)")
