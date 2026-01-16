"""
Pipeline Integration - Connects CV tracking output with Wyscout events

This module bridges:
1. Wyscout event data (from cloud bucket)
2. Tracking data (from your CV pipeline)
3. Decision analysis (event_tracking_sync)

Usage:
    from pipeline_integration import MatchAnalysisPipeline

    pipeline = MatchAnalysisPipeline(bucket_name="your-bucket")
    results = pipeline.analyze_match(
        match_id="12345",
        tracking_data=frame_records  # from your CV pipeline
    )
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Import sync module
import sys
sys.path.insert(0, str(Path(__file__).parent))

from event_tracking_sync import (
    EventTrackingSync,
    DecisionAnalyzer,
    KeyMomentsExtractor,
    print_analysis_report,
)


@dataclass
class BucketConfig:
    """Configuration for cloud bucket access."""
    bucket_name: str
    provider: str = "gcs"  # "gcs" or "s3"
    credentials_path: Optional[str] = None
    wyscout_prefix: str = "wyscout/events/"  # Path prefix in bucket


class WyscoutBucketLoader:
    """
    Loads Wyscout event data from cloud storage bucket.

    Supports:
    - Google Cloud Storage (GCS)
    - AWS S3
    - Local filesystem (for testing)
    """

    def __init__(self, config: BucketConfig):
        self.config = config
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize cloud storage client."""
        if self.config.provider == "gcs":
            self._init_gcs()
        elif self.config.provider == "s3":
            self._init_s3()
        elif self.config.provider == "local":
            # Local filesystem - no client needed
            pass
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _init_gcs(self):
        """Initialize Google Cloud Storage client."""
        try:
            from google.cloud import storage

            if self.config.credentials_path:
                self.client = storage.Client.from_service_account_json(
                    self.config.credentials_path
                )
            else:
                # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
                self.client = storage.Client()

            self.bucket = self.client.bucket(self.config.bucket_name)
            print(f"Connected to GCS bucket: {self.config.bucket_name}")

        except ImportError:
            print("google-cloud-storage not installed. Install with:")
            print("  pip install google-cloud-storage")
            raise
        except Exception as e:
            print(f"Failed to connect to GCS: {e}")
            raise

    def _init_s3(self):
        """Initialize AWS S3 client."""
        try:
            import boto3

            if self.config.credentials_path:
                # Load credentials from file
                self.client = boto3.client('s3')
            else:
                # Use default AWS credentials
                self.client = boto3.client('s3')

            print(f"Connected to S3 bucket: {self.config.bucket_name}")

        except ImportError:
            print("boto3 not installed. Install with:")
            print("  pip install boto3")
            raise

    def list_matches(self) -> List[str]:
        """List available match IDs in the bucket."""
        matches = []

        if self.config.provider == "gcs":
            blobs = self.bucket.list_blobs(prefix=self.config.wyscout_prefix)
            for blob in blobs:
                # Extract match ID from path like "wyscout/events/12345.json"
                name = blob.name.replace(self.config.wyscout_prefix, "")
                if name.endswith(".json"):
                    matches.append(name.replace(".json", ""))

        elif self.config.provider == "s3":
            response = self.client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=self.config.wyscout_prefix
            )
            for obj in response.get('Contents', []):
                name = obj['Key'].replace(self.config.wyscout_prefix, "")
                if name.endswith(".json"):
                    matches.append(name.replace(".json", ""))

        elif self.config.provider == "local":
            local_path = Path(self.config.bucket_name) / self.config.wyscout_prefix
            if local_path.exists():
                for f in local_path.glob("*.json"):
                    matches.append(f.stem)

        return matches

    def load_events(self, match_id: str) -> List[Dict]:
        """Load Wyscout events for a specific match."""
        blob_path = f"{self.config.wyscout_prefix}{match_id}.json"

        if self.config.provider == "gcs":
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                raise FileNotFoundError(f"Match not found: {match_id}")

            content = blob.download_as_text()
            data = json.loads(content)

        elif self.config.provider == "s3":
            response = self.client.get_object(
                Bucket=self.config.bucket_name,
                Key=blob_path
            )
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)

        elif self.config.provider == "local":
            file_path = Path(self.config.bucket_name) / blob_path
            if not file_path.exists():
                raise FileNotFoundError(f"Match not found: {match_id}")
            with open(file_path, 'r') as f:
                data = json.load(f)

        # Handle various Wyscout formats
        if isinstance(data, list):
            events = data
        else:
            events = data.get('events', data.get('data', []))

        print(f"Loaded {len(events)} events for match {match_id}")
        return events


class MatchAnalysisPipeline:
    """
    Complete pipeline for analyzing match decisions.

    Connects:
    1. Wyscout events (from bucket)
    2. Tracking data (from CV pipeline)
    3. Decision analysis
    """

    def __init__(
        self,
        bucket_name: str,
        provider: str = "gcs",
        credentials_path: Optional[str] = None,
        wyscout_prefix: str = "wyscout/events/",
    ):
        self.config = BucketConfig(
            bucket_name=bucket_name,
            provider=provider,
            credentials_path=credentials_path,
            wyscout_prefix=wyscout_prefix,
        )
        self.wyscout_loader = WyscoutBucketLoader(self.config)
        self.analyzer = DecisionAnalyzer()

    def analyze_match(
        self,
        match_id: str,
        tracking_data: List[Dict],
        video_fps: float = 25.0,
        period: int = 1,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Analyze a match by combining Wyscout events with tracking data.

        Args:
            match_id: Wyscout match ID to fetch events for
            tracking_data: Frame records from your CV pipeline
                          (output of FootballTracker.process_video() or data_export)
            video_fps: Frames per second of the video (for timestamp calculation)
            period: Match period (1 or 2)
            output_path: Optional path to save analysis JSON

        Returns:
            Complete analysis results
        """
        # Load Wyscout events from bucket
        events = self.wyscout_loader.load_events(match_id)

        # Convert tracking data to sync format
        tracking_frames = self._convert_tracking_format(tracking_data, period)

        # Synchronize
        sync = EventTrackingSync()
        sync.load_wyscout_events(events)
        sync.load_tracking_data(tracking_frames)
        synced_events = sync.synchronize()

        # Analyze
        match_analysis = self.analyzer.analyze_match(synced_events)

        # Extract key moments
        extractor = KeyMomentsExtractor(self.analyzer)
        key_moments = extractor.extract_key_moments(synced_events)

        results = {
            'match_id': match_id,
            'summary': {
                'total_events': match_analysis['total_events'],
                'events_with_context': match_analysis['events_with_context'],
                'avg_decision_quality': match_analysis['avg_decision_quality'],
                'optimal_decisions': match_analysis['optimal_decisions'],
                'suboptimal_decisions': match_analysis['suboptimal_decisions'],
            },
            'key_moments': key_moments,
            'detailed_analyses': match_analysis['analyses'][:100],
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")

        return results

    def _convert_tracking_format(
        self,
        frame_records: List[Dict],
        period: int = 1
    ) -> List[Dict]:
        """
        Convert CV pipeline output to sync module format.

        The CV pipeline outputs (from data_export.py):
        {
            "frame": int,
            "timestamp": float,
            "players": [{"track_id", "team", "x", "y", ...}],
            "ball": {"x", "y", ...}
        }

        The sync module expects (already compatible, but we add period):
        {
            "timestamp": float,
            "period": int,
            "players": [{"playerId", "team", "x", "y", ...}],
            "ball": {...}
        }
        """
        converted = []

        for frame in frame_records:
            tracking_frame = {
                'timestamp': frame.get('timestamp', frame.get('frame', 0) / 25.0),
                'period': period,
                'players': [],
                'ball': frame.get('ball'),
            }

            for player in frame.get('players', []):
                tracking_frame['players'].append({
                    'playerId': str(player.get('track_id', player.get('id', ''))),
                    'team': player.get('team', 'unknown'),
                    'x': player.get('x', 0),
                    'y': player.get('y', 0),
                    'jersey_number': player.get('jersey_number'),
                })

            converted.append(tracking_frame)

        return converted

    def analyze_from_files(
        self,
        match_id: str,
        tracking_json_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Analyze using a tracking JSON file instead of live data.

        Args:
            match_id: Wyscout match ID
            tracking_json_path: Path to exported tracking JSON
            output_path: Optional output path
        """
        with open(tracking_json_path, 'r') as f:
            tracking_data = json.load(f)

        # Handle wrapped format
        if isinstance(tracking_data, dict):
            tracking_data = tracking_data.get('frames', tracking_data.get('data', []))

        return self.analyze_match(match_id, tracking_data, output_path=output_path)


def analyze_from_cv_pipeline(
    match_id: str,
    bucket_name: str,
    tracking_data: List[Dict],
    provider: str = "gcs",
    output_path: Optional[str] = None,
) -> Dict:
    """
    Convenience function to analyze a match directly from CV pipeline output.

    Example:
        from src.main import FootballTracker
        from src.decision_engine.pipeline_integration import analyze_from_cv_pipeline

        # Run tracking
        tracker = FootballTracker(config)
        frame_records = tracker.process_video("match.mp4")

        # Analyze decisions
        results = analyze_from_cv_pipeline(
            match_id="wyscout_12345",
            bucket_name="your-bucket",
            tracking_data=frame_records,
        )
    """
    pipeline = MatchAnalysisPipeline(
        bucket_name=bucket_name,
        provider=provider,
    )
    return pipeline.analyze_match(match_id, tracking_data, output_path=output_path)


# Demo/test with local files
def demo_local():
    """Demo using local files instead of cloud bucket."""
    # Create test data directory
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    events_dir = test_dir / "wyscout" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    # Write sample Wyscout events
    sample_events = [
        {
            "eventId": 1,
            "eventName": "Pass",
            "subEventName": "Through pass",
            "eventSec": 45.0,
            "matchPeriod": "1H",
            "playerId": "p10",
            "teamId": "home",
            "positions": [{"x": 50, "y": 50}, {"x": 70, "y": 45}],
            "tags": [{"name": "accurate"}]
        },
        {
            "eventId": 2,
            "eventName": "Shot",
            "subEventName": "Shot",
            "eventSec": 48.0,
            "matchPeriod": "1H",
            "playerId": "p9",
            "teamId": "home",
            "positions": [{"x": 90, "y": 48}],
            "tags": [{"name": "Goal"}]
        }
    ]

    with open(events_dir / "demo_match.json", 'w') as f:
        json.dump(sample_events, f)

    # Sample tracking data (simulating CV pipeline output)
    sample_tracking = [
        {
            "frame": 0,
            "timestamp": 44.9,
            "players": [
                {"track_id": 10, "team": "home", "x": 52.5, "y": 34.0},
                {"track_id": 9, "team": "home", "x": 73.5, "y": 30.6},
                {"track_id": 7, "team": "home", "x": 60.0, "y": 40.0},
                {"track_id": 101, "team": "away", "x": 65.0, "y": 35.0},
                {"track_id": 102, "team": "away", "x": 80.0, "y": 32.0},
            ],
            "ball": {"x": 52.5, "y": 34.0}
        },
        {
            "frame": 75,
            "timestamp": 47.9,
            "players": [
                {"track_id": 10, "team": "home", "x": 58.0, "y": 36.0},
                {"track_id": 9, "team": "home", "x": 94.5, "y": 32.6},
                {"track_id": 7, "team": "home", "x": 75.0, "y": 38.0},
                {"track_id": 101, "team": "away", "x": 85.0, "y": 34.0},
                {"track_id": 102, "team": "away", "x": 98.0, "y": 36.0},
            ],
            "ball": {"x": 94.5, "y": 32.6}
        }
    ]

    # Run analysis with local provider
    pipeline = MatchAnalysisPipeline(
        bucket_name=str(test_dir),
        provider="local",
        wyscout_prefix="wyscout/events/",
    )

    results = pipeline.analyze_match(
        match_id="demo_match",
        tracking_data=sample_tracking,
    )

    print_analysis_report(results)

    return results


if __name__ == '__main__':
    demo_local()
