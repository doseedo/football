"""Example usage of the tactical analysis GNN system."""

import torch
from typing import List, Dict
import numpy as np

from .graph_builder import TrackingGraphBuilder
from .gnn_model import create_tactical_gnn
from .team_state import TeamStateClassifier, TacticalState
from .tactical_metrics import TacticalMetricsCalculator


def example_basic_usage():
    """Example: Basic tactical analysis from tracking data."""

    # Sample tracking data (single frame)
    frame_data = {
        'frame': 0,
        'timestamp': 0.0,
        'players': [
            {'track_id': 1, 'x': -20.0, 'y': 0.0, 'vx': 0.5, 'vy': 0.1, 'team': 0},
            {'track_id': 2, 'x': -15.0, 'y': 5.0, 'vx': 1.0, 'vy': 0.0, 'team': 0},
            {'track_id': 3, 'x': -10.0, 'y': -5.0, 'vx': 0.8, 'vy': 0.2, 'team': 0},
            {'track_id': 4, 'x': 0.0, 'y': 0.0, 'vx': 1.2, 'vy': 0.0, 'team': 0},
            {'track_id': 5, 'x': 10.0, 'y': 3.0, 'vx': 0.5, 'vy': -0.1, 'team': 0},
            {'track_id': 11, 'x': 20.0, 'y': 0.0, 'vx': -0.5, 'vy': 0.0, 'team': 1},
            {'track_id': 12, 'x': 15.0, 'y': -5.0, 'vx': -1.0, 'vy': 0.1, 'team': 1},
            {'track_id': 13, 'x': 10.0, 'y': 5.0, 'vx': -0.8, 'vy': 0.0, 'team': 1},
            {'track_id': 14, 'x': 5.0, 'y': 0.0, 'vx': -1.2, 'vy': 0.0, 'team': 1},
            {'track_id': 15, 'x': -10.0, 'y': 2.0, 'vx': -0.5, 'vy': 0.0, 'team': 1},
        ],
        'ball': {'x': 5.0, 'y': 1.0}
    }

    # Step 1: Convert tracking data to graph
    print("Step 1: Building graph from tracking data...")
    graph_builder = TrackingGraphBuilder(
        proximity_threshold=15.0,
        edge_type='proximity'
    )
    graph = graph_builder.build_graph(frame_data)

    print(f"  Graph nodes: {graph.num_nodes}")
    print(f"  Graph edges: {graph.edge_index.shape[1]}")
    print(f"  Node features: {graph.x.shape}")
    print(f"  Edge features: {graph.edge_attr.shape}")

    # Step 2: Create GNN model
    print("\nStep 2: Creating GNN model...")
    gnn_model = create_tactical_gnn({
        'input_dim': 12,
        'hidden_dim': 128,
        'num_layers': 4,
        'output_dim': 64,
        'gnn_type': 'gat',
        'num_heads': 4
    })

    print(f"  Model parameters: {sum(p.numel() for p in gnn_model.parameters()):,}")

    # Step 3: Get GNN embeddings
    print("\nStep 3: Computing GNN embeddings...")
    gnn_model.eval()
    with torch.no_grad():
        node_embeddings, graph_embedding = gnn_model(graph)

    print(f"  Node embeddings shape: {node_embeddings.shape}")
    print(f"  Graph embedding shape: {graph_embedding.shape}")

    # Step 4: Classify team state
    print("\nStep 4: Classifying tactical state...")
    state_classifier = TeamStateClassifier(
        gnn_model=gnn_model,
        use_gnn_classifier=False  # Use rule-based for demo
    )

    team_state = state_classifier.classify(graph, timestamp=0.0)

    print(f"  Tactical state: {team_state.state.name}")
    print(f"  Confidence: {team_state.confidence:.3f}")
    print(f"  Possession team: {team_state.possession_team}")
    print(f"  Possession probabilities: {team_state.possession_probability}")
    print(f"  Pressing intensity: {team_state.pressing_intensity:.3f}")
    print(f"  Defensive line height: {team_state.defensive_line_height:.2f}m")
    print(f"  Team compactness: {team_state.team_compactness:.2f}m")

    # Step 5: Calculate tactical metrics
    print("\nStep 5: Computing tactical metrics...")
    metrics_calculator = TacticalMetricsCalculator()
    tactical_metrics = metrics_calculator.calculate_all_metrics(
        graph=graph,
        ball_position=(5.0, 1.0),
        possession_team=team_state.possession_team
    )

    print(f"  Space control ratio: {tactical_metrics.space_control_ratio:.3f}")
    print(f"  Pressure on ball: {tactical_metrics.pressure_on_ball:.3f}")
    print(f"  Passing lane count: {tactical_metrics.passing_lane_count}")
    print(f"  Passing lane quality: {tactical_metrics.passing_lane_quality:.3f}")
    print(f"  Progressive pass options: {tactical_metrics.progressive_pass_options}")
    print(f"  Expected threat (xT): {tactical_metrics.expected_threat:.4f}")
    print(f"  Pitch control: {tactical_metrics.pitch_control_possession:.3f}")


def example_temporal_analysis():
    """Example: Temporal graph analysis over multiple frames."""

    print("\n" + "="*60)
    print("EXAMPLE: Temporal Graph Analysis")
    print("="*60)

    # Generate sequence of frames
    frame_sequence = []
    for t in range(10):
        frame = {
            'timestamp': t * 0.04,  # 25 fps
            'players': [
                {'track_id': i, 'x': -20 + i*5 + t*0.5, 'y': np.sin(t/5 + i) * 10,
                 'vx': 0.5, 'vy': 0.1, 'team': 0 if i < 5 else 1}
                for i in range(10)
            ],
            'ball': {'x': 0 + t*0.8, 'y': np.sin(t/3) * 5}
        }
        frame_sequence.append(frame)

    # Build temporal graph
    graph_builder = TrackingGraphBuilder()
    temporal_graph = graph_builder.build_temporal_graph(frame_sequence)

    print(f"\nTemporal graph with {len(temporal_graph.graphs)} frames")
    print(f"Time span: {temporal_graph.timestamps[-1] - temporal_graph.timestamps[0]:.2f}s")

    # Analyze state transitions
    gnn_model = create_tactical_gnn()
    state_classifier = TeamStateClassifier(gnn_model, use_gnn_classifier=False)

    states = []
    for i, graph in enumerate(temporal_graph.graphs):
        team_state = state_classifier.classify(graph, timestamp=temporal_graph.timestamps[i])
        states.append(team_state.state)

    print("\nState sequence:")
    for i, state in enumerate(states):
        print(f"  Frame {i}: {state.name}")

    # Detect state transitions
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i-1]:
            transitions.append((i, states[i-1].name, states[i].name))

    print(f"\nDetected {len(transitions)} state transitions:")
    for frame, from_state, to_state in transitions:
        print(f"  Frame {frame}: {from_state} → {to_state}")


def example_integration_pipeline():
    """Example: Complete integration pipeline."""

    print("\n" + "="*60)
    print("EXAMPLE: Complete Integration Pipeline")
    print("="*60)

    # Simulated tracking data from main pipeline
    tracking_results = {
        'frames': [
            {
                'frame': i,
                'timestamp': i * 0.04,
                'players': [
                    {
                        'track_id': j,
                        'x': np.random.uniform(-40, 40),
                        'y': np.random.uniform(-30, 30),
                        'vx': np.random.uniform(-2, 2),
                        'vy': np.random.uniform(-2, 2),
                        'team': 0 if j < 6 else 1,
                        'speed': np.random.uniform(0, 8),
                        'jersey_number': j + 1
                    }
                    for j in range(12)
                ],
                'ball': {
                    'x': np.random.uniform(-20, 20),
                    'y': np.random.uniform(-15, 15)
                }
            }
            for i in range(5)
        ]
    }

    # Initialize components
    graph_builder = TrackingGraphBuilder()
    gnn_model = create_tactical_gnn()
    state_classifier = TeamStateClassifier(gnn_model, use_gnn_classifier=False)
    metrics_calculator = TacticalMetricsCalculator()

    # Process each frame
    results = []
    for frame_data in tracking_results['frames']:
        # Build graph
        graph = graph_builder.build_graph(frame_data)

        # Classify state
        team_state = state_classifier.classify(graph, timestamp=frame_data['timestamp'])

        # Calculate metrics
        ball_pos = (frame_data['ball']['x'], frame_data['ball']['y'])
        tactical_metrics = metrics_calculator.calculate_all_metrics(
            graph, ball_pos, team_state.possession_team
        )

        # Store results
        result = {
            'frame': frame_data['frame'],
            'timestamp': frame_data['timestamp'],
            'tactical_state': team_state.state.name,
            'confidence': team_state.confidence,
            'possession_team': team_state.possession_team,
            'pressing_intensity': team_state.pressing_intensity,
            'space_control': tactical_metrics.space_control_ratio,
            'passing_lanes': tactical_metrics.passing_lane_count,
            'expected_threat': tactical_metrics.expected_threat,
            'defensive_line': tactical_metrics.defensive_line_height,
            'team_compactness': tactical_metrics.team_compactness,
        }
        results.append(result)

    # Display results
    print("\nTactical Analysis Results:")
    print("-" * 60)
    for r in results:
        print(f"Frame {r['frame']} (t={r['timestamp']:.2f}s):")
        print(f"  State: {r['tactical_state']:20s} (conf: {r['confidence']:.2f})")
        print(f"  Possession: Team {r['possession_team']}")
        print(f"  Pressing: {r['pressing_intensity']:.2f} | Space: {r['space_control']:.2f}")
        print(f"  Lanes: {r['passing_lanes']:2d} | xT: {r['expected_threat']:.3f}")
        print()

    return results


def export_tactical_data(results: List[Dict], output_path: str = 'tactical_output.json'):
    """Export tactical analysis results."""
    import json

    print(f"\nExporting results to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("Export complete!")


if __name__ == '__main__':
    print("="*60)
    print("Graph Neural Network Tactical Analysis - Examples")
    print("="*60)

    # Run examples
    example_basic_usage()
    example_temporal_analysis()
    results = example_integration_pipeline()

    # Export results (optional)
    # export_tactical_data(results)

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
