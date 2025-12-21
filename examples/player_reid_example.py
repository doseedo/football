#!/usr/bin/env python
"""Example script demonstrating the player re-identification system.

This script shows how to use the complete pipeline for player identification
without prior training on the match.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from pathlib import Path

from src.identity import (
    create_player_identifier,
    visualize_identity,
    PlayerIdentity
)


def load_sample_data():
    """
    Load sample player crops and track IDs.

    In a real application, these would come from your detector/tracker.
    """
    # Simulating detections
    num_players = 10
    player_crops = []

    for i in range(num_players):
        # Create dummy player crops (replace with real detections)
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        # Simulate jersey colors (2 teams)
        if i < 5:  # Team A - Red
            crop[50:150, 30:100] = [200, 50, 50]
        else:  # Team B - Blue
            crop[50:150, 30:100] = [50, 50, 200]

        player_crops.append(crop)

    player_crops = np.array(player_crops)
    track_ids = np.arange(num_players)

    return player_crops, track_ids


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Player Identification")
    print("=" * 60)

    # Initialize identifier (without pretrained weights for demo)
    identifier = create_player_identifier({
        'device': 'cuda',
        'num_teams': 3  # 2 teams + referee
    })

    # Load sample data
    player_crops, track_ids = load_sample_data()

    # Process frame
    identities = identifier.process_frame(player_crops, track_ids)

    # Print results
    print(f"\nIdentified {len(identities)} players:")
    for identity in identities:
        print(f"  Track {identity.track_id} -> Player {identity.stable_id}")
        print(f"    Team: {identity.team_id}")
        if identity.jersey_number:
            print(f"    Jersey: #{identity.jersey_number}")
        print(f"    Confidence: {identity.confidence:.2f}")
        print()


def example_multi_frame():
    """Multi-frame processing with re-identification."""
    print("=" * 60)
    print("Example 2: Multi-Frame Processing")
    print("=" * 60)

    identifier = create_player_identifier({'device': 'cuda'})

    # Simulate multiple frames
    num_frames = 5
    for frame_idx in range(num_frames):
        print(f"\n--- Frame {frame_idx + 1} ---")

        # Get detections (simulated)
        player_crops, track_ids = load_sample_data()

        # Process frame
        identities = identifier.process_frame(player_crops, track_ids)

        # Print stable IDs
        print(f"Detected {len(identities)} players:")
        for identity in identities:
            print(f"  Track {identity.track_id} -> Stable ID {identity.stable_id}")

    # Export final roster
    print("\n" + "=" * 60)
    print("Final Match Roster:")
    print("=" * 60)
    roster = identifier.export_roster()

    for player_id, info in sorted(roster.items()):
        team_name = ['Team A', 'Team B', 'Referee'][info['team_id']]
        jersey = f"#{info['jersey_number']}" if info['jersey_number'] else "N/A"
        print(f"Player {player_id}: {team_name}, Jersey {jersey}")


def example_with_pretrained():
    """Example with pretrained OSNet weights."""
    print("=" * 60)
    print("Example 3: Using Pretrained OSNet")
    print("=" * 60)

    # Check if pretrained weights exist
    osnet_path = 'models/osnet_ain_x1_0.pth'

    if not Path(osnet_path).exists():
        print(f"\nPretrained weights not found at {osnet_path}")
        print("To use pretrained OSNet, download weights from:")
        print("https://github.com/KaiyangZhou/deep-person-reid")
        print("\nUsing random initialization for demonstration.")
        osnet_path = None

    # Initialize with OSNet
    identifier = create_player_identifier({
        'osnet_path': osnet_path,
        'device': 'cuda',
        'num_teams': 3
    })

    # Process sample data
    player_crops, track_ids = load_sample_data()
    identities = identifier.process_frame(player_crops, track_ids)

    print(f"\nProcessed {len(identities)} players with appearance embeddings")

    # Get team information
    team_info = identifier.get_team_info()
    print(f"\nTeam Classification:")
    print(f"  Fitted: {team_info['is_fitted']}")
    print(f"  Num Teams: {team_info['num_teams']}")
    if team_info['referee_id'] is not None:
        print(f"  Referee ID: {team_info['referee_id']}")


def example_visualization():
    """Example with visualization."""
    print("=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)

    identifier = create_player_identifier({'device': 'cuda'})

    # Process sample data
    player_crops, track_ids = load_sample_data()
    identities = identifier.process_frame(player_crops, track_ids)

    # Create visualization
    print("\nGenerating visualizations...")

    output_dir = Path('output/reid_examples')
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (identity, crop) in enumerate(zip(identities, player_crops)):
        vis = visualize_identity(crop, identity)

        # Save visualization
        output_path = output_dir / f'player_{identity.stable_id}_track_{identity.track_id}.jpg'
        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Saved visualizations to {output_dir}")


def example_temporal_aggregation():
    """Example showing temporal aggregation for jersey numbers."""
    print("=" * 60)
    print("Example 5: Temporal Jersey Number Aggregation")
    print("=" * 60)

    from src.identity import TemporalNumberAggregator

    aggregator = TemporalNumberAggregator(window_size=30, min_confidence=0.5)

    # Simulate detections over multiple frames
    track_id = 1
    predictions = [
        (10, 0.8),  # Frame 1: #10 with 0.8 confidence
        (10, 0.7),  # Frame 2: #10 with 0.7 confidence
        (None, 0.0),  # Frame 3: No detection
        (10, 0.9),  # Frame 4: #10 with 0.9 confidence
        (11, 0.5),  # Frame 5: #11 with 0.5 confidence (noise)
        (10, 0.8),  # Frame 6: #10 with 0.8 confidence
    ]

    print(f"\nProcessing predictions for track {track_id}:")
    for frame_idx, (number, conf) in enumerate(predictions, 1):
        aggregator.add_prediction(track_id, number, conf)

        stable_number = aggregator.get_stable_number(track_id, min_votes=3)

        print(f"  Frame {frame_idx}: Detected #{number} (conf={conf:.2f}) -> "
              f"Stable: #{stable_number if stable_number else 'N/A'}")

    # Final result
    final_number = aggregator.get_stable_number(track_id, min_votes=3)
    final_confidence = aggregator.get_confidence(track_id, final_number) if final_number else 0.0

    print(f"\nFinal assignment: #{final_number} (confidence={final_confidence:.2f})")


def main():
    """Run all examples."""
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Multi-Frame", example_multi_frame),
        ("Pretrained Weights", example_with_pretrained),
        ("Visualization", example_visualization),
        ("Temporal Aggregation", example_temporal_aggregation),
    ]

    print("\n" + "=" * 60)
    print("Player Re-Identification System - Examples")
    print("=" * 60)

    for name, func in examples:
        try:
            func()
            print(f"\n✓ {name} completed successfully\n")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}\n")

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
