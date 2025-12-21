"""Integration test for tactical analysis module."""

import torch
import numpy as np
from graph_builder import TrackingGraphBuilder
from gnn_model import create_tactical_gnn
from team_state import TeamStateClassifier, TacticalState
from tactical_metrics import TacticalMetricsCalculator


def test_basic_integration():
    """Test basic integration of all components."""

    print("Testing Tactical Analysis Integration...")
    print("-" * 60)

    # Sample frame data
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

    # Test 1: Graph construction
    print("\n1. Testing graph construction...")
    graph_builder = TrackingGraphBuilder()
    graph = graph_builder.build_graph(frame_data)

    assert graph.num_nodes == 10, f"Expected 10 nodes, got {graph.num_nodes}"
    assert graph.x.shape[1] == 12, f"Expected 12 features, got {graph.x.shape[1]}"
    print(f"   ✓ Graph created: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    # Test 2: GNN model
    print("\n2. Testing GNN model...")
    gnn_model = create_tactical_gnn()
    gnn_model.eval()

    with torch.no_grad():
        node_embeddings, graph_embedding = gnn_model(graph)

    assert node_embeddings.shape[0] == 10, "Node embeddings size mismatch"
    assert graph_embedding.shape[0] == 64, "Graph embedding size mismatch"
    print(f"   ✓ GNN forward pass: node_emb {node_embeddings.shape}, graph_emb {graph_embedding.shape}")

    # Test 3: Team state classification
    print("\n3. Testing team state classification...")
    state_classifier = TeamStateClassifier(gnn_model, use_gnn_classifier=False)
    team_state = state_classifier.classify(graph)

    assert isinstance(team_state.state, TacticalState), "Invalid state type"
    assert 0 <= team_state.confidence <= 1, "Confidence out of range"
    assert team_state.possession_team in [0, 1], "Invalid possession team"
    print(f"   ✓ State: {team_state.state.name} (conf: {team_state.confidence:.2f})")
    print(f"   ✓ Possession: Team {team_state.possession_team} ({team_state.possession_probability[team_state.possession_team]:.2f})")
    print(f"   ✓ Pressing: {team_state.pressing_intensity:.2f}")

    # Test 4: Tactical metrics
    print("\n4. Testing tactical metrics...")
    metrics_calc = TacticalMetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(
        graph,
        ball_position=(5.0, 1.0),
        possession_team=team_state.possession_team
    )

    assert 0 <= metrics.space_control_ratio <= 1, "Space control out of range"
    assert metrics.passing_lane_count >= 0, "Invalid passing lane count"
    assert 0 <= metrics.expected_threat <= 1, "xT out of range"
    print(f"   ✓ Space control: {metrics.space_control_ratio:.2f}")
    print(f"   ✓ Passing lanes: {metrics.passing_lane_count} (quality: {metrics.passing_lane_quality:.2f})")
    print(f"   ✓ Expected threat: {metrics.expected_threat:.3f}")
    print(f"   ✓ Pressure on ball: {metrics.pressure_on_ball:.2f}")

    # Test 5: Temporal analysis
    print("\n5. Testing temporal graph...")
    frame_sequence = [frame_data for _ in range(5)]  # Same frame repeated
    temporal_graph = graph_builder.build_temporal_graph(frame_sequence)

    assert len(temporal_graph.graphs) == 5, "Wrong number of temporal graphs"
    print(f"   ✓ Temporal graph: {len(temporal_graph.graphs)} frames")

    print("\n" + "="*60)
    print("All integration tests passed! ✓")
    print("="*60)

    return True


if __name__ == '__main__':
    try:
        success = test_basic_integration()
        if success:
            print("\n✓ Integration test successful!")
            exit(0)
        else:
            print("\n✗ Integration test failed!")
            exit(1)
    except Exception as e:
        print(f"\n✗ Integration test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
