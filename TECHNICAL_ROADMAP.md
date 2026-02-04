# Technical Implementation Roadmap

## Overview

This document provides a concrete, step-by-step technical roadmap for building the football AI system. It specifies exact tools, libraries, data sources, and implementation paths.

---

## Part 1: The Technology Stack

### Core Python Libraries

| Library | Purpose | Installation | Docs |
|---------|---------|--------------|------|
| **kloppy** | Load data from any provider | `pip install kloppy` | [kloppy.pysport.org](https://kloppy.pysport.org/) |
| **socceraction** | VAEP/xT implementation | `pip install socceraction` | [socceraction.readthedocs.io](https://socceraction.readthedocs.io/) |
| **mplsoccer** | Pitch visualization | `pip install mplsoccer` | [mplsoccer.readthedocs.io](https://mplsoccer.readthedocs.io/) |
| **PyTorch** | Deep learning | `pip install torch` | [pytorch.org](https://pytorch.org/) |
| **PyTorch Geometric** | Graph neural networks | `pip install torch_geometric` | [pyg.org](https://pyg.org/) |
| **ultralytics** | YOLO detection | `pip install ultralytics` | [docs.ultralytics.com](https://docs.ultralytics.com/) |

### Data Sources

| Source | Type | Access | Use Case |
|--------|------|--------|----------|
| **StatsBomb Open Data** | Event data | Free | Training, validation, benchmarking |
| **Wyscout** | Event data | Subscription ($) | Production analysis |
| **SkillCorner** | Tracking from broadcast | Subscription ($$) | Tracking without stadium hardware |
| **Metrica Sample Data** | Tracking + events | Free | Development, testing |

### Infrastructure

| Component | Tool | Notes |
|-----------|------|-------|
| Version control | Git | Already set up |
| ML experiments | MLflow or Weights & Biases | Track experiments |
| Compute | GPU (local or cloud) | For training GNNs |
| Database | PostgreSQL or SQLite | Store processed data |
| API | FastAPI | Serve predictions |
| Frontend | Streamlit or React | Visualization |

---

## Part 2: Data Loading Pipeline

### Step 1: Set Up kloppy

```python
# Install
pip install kloppy

# Load StatsBomb open data
from kloppy import statsbomb

# Load a match
dataset = statsbomb.load_open_data(
    match_id=3788741,  # Example: 2022 World Cup Final
    coordinates="statsbomb"
)

# Convert to DataFrame
df = dataset.to_df()

# Access events
for event in dataset.events:
    print(f"{event.player}: {event.event_type} at {event.coordinates}")
```

### Step 2: Load Tracking Data (Metrica Sample)

```python
from kloppy import metrica

# Load sample tracking data
dataset = metrica.load_open_data(
    match_id=1,
    sample_rate=1/25  # 25 fps
)

# Access frames
for frame in dataset.frames:
    print(f"Frame {frame.frame_id}: Ball at {frame.ball_coordinates}")
    for player in frame.players_data:
        print(f"  {player.player}: {player.coordinates}")
```

### Step 3: Standardize Coordinates

```python
from kloppy.domain import Orientation, Provider
from kloppy import statsbomb

# Load with specific coordinate system
dataset = statsbomb.load_open_data(
    match_id=3788741,
    coordinates="secondspectrum",  # Standardize to SecondSpectrum coords
)

# Or transform after loading
dataset = dataset.transform(
    to_pitch_dimensions=[105, 68],  # Standard pitch size
    to_orientation=Orientation.HOME_TEAM,
)
```

---

## Part 3: Implement VAEP (Action Valuation)

### Step 1: Install socceraction

```python
pip install socceraction
```

### Step 2: Load and Convert Data to SPADL

```python
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader

# Load StatsBomb data
loader = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})

# Get competition and matches
competitions = loader.competitions()
matches = loader.matches(competition_id=43, season_id=3)  # World Cup 2018

# Convert to SPADL format
for match in matches:
    events = loader.events(match_id=match['match_id'])
    actions = spadl.statsbomb.convert_to_actions(events, match['home_team_id'])
    # actions is now a DataFrame in SPADL format
```

### Step 3: Compute VAEP Values

```python
from socceraction.vaep import VAEP
from socceraction.vaep.features import gamestates as fs_gamestates
from socceraction.vaep.labels import scores, concedes

# Initialize VAEP model
vaep_model = VAEP()

# Generate features and labels for training
X = fs_gamestates(actions)
Y_scores = scores(actions)
Y_concedes = concedes(actions)

# Train the model
vaep_model.fit(X, Y_scores, Y_concedes)

# Predict action values
predictions = vaep_model.predict(X)
action_values = predictions['offensive_value'] - predictions['defensive_value']
```

### Step 4: Create Player Ratings

```python
# Add values to actions DataFrame
actions['vaep_value'] = action_values

# Aggregate by player
player_ratings = actions.groupby('player_id').agg({
    'vaep_value': ['sum', 'mean', 'count']
})

# Normalize by minutes played
player_ratings['vaep_per_90'] = player_ratings['vaep_value']['sum'] / (minutes_played / 90)
```

---

## Part 4: Build Graph Neural Network (TacticAI Approach)

### Step 1: Install PyTorch Geometric

```bash
pip install torch torch_geometric
```

### Step 2: Create Graph from Game State

```python
import torch
from torch_geometric.data import Data

def game_state_to_graph(frame, ball_position):
    """
    Convert a tracking frame to a graph.

    Nodes: players + ball
    Edges: spatial relationships
    """
    # Node features: [x, y, vx, vy, team, is_ball]
    node_features = []

    # Add players
    for player in frame.players:
        node_features.append([
            player.x, player.y,
            player.vx, player.vy,
            1.0 if player.team == 'home' else 0.0,
            0.0  # not ball
        ])

    # Add ball
    node_features.append([
        ball_position.x, ball_position.y,
        ball_position.vx, ball_position.vy,
        0.5,  # neutral
        1.0   # is ball
    ])

    x = torch.tensor(node_features, dtype=torch.float)

    # Create edges: fully connected for simplicity
    # (can be refined to spatial proximity)
    num_nodes = len(node_features)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
```

### Step 3: Define GNN Model

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class TacticalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # Global pooling
        x = global_mean_pool(x, batch)

        # Prediction
        return self.lin(x)

# Initialize
model = TacticalGNN(in_channels=6, hidden_channels=64, out_channels=1)
```

### Step 4: Train for Outcome Prediction

```python
from torch_geometric.loader import DataLoader

# Assume we have a list of graph-label pairs
# graphs: list of Data objects
# labels: tensor of outcomes (e.g., 1 = goal, 0 = no goal)

dataset = [(g, l) for g, l in zip(graphs, labels)]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    for data, label in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), label.float())
        loss.backward()
        optimizer.step()
```

---

## Part 5: Integrate Your Existing Decision Engine

### Current Architecture (from codebase)

Your existing code has:
- `src/decision_engine/elimination.py` — Elimination calculator
- `src/decision_engine/defense_physics.py` — Force-based defense model
- `src/decision_engine/state_scoring.py` — Game state evaluator
- `src/decision_engine/block_models.py` — Defensive block templates

### Integration Points

```python
# 1. Use physics features as input to learned models

from decision_engine.elimination import EliminationCalculator
from decision_engine.state_scoring import GameStateEvaluator

def extract_physics_features(game_state):
    """
    Use existing physics engine to generate features.
    """
    elim_calc = EliminationCalculator()
    evaluator = GameStateEvaluator()

    # Get physics-based features
    elimination_state = elim_calc.calculate(game_state)
    state_score = evaluator.evaluate(game_state)

    return {
        'eliminated_count': elimination_state.eliminated_count,
        'elimination_ratio': elimination_state.elimination_ratio,
        'position_score': state_score.total,
        'proximity_score': state_score.proximity,
        'angle_score': state_score.angle,
        'density_score': state_score.density,
    }

# 2. Combine with learned features

def combined_features(game_state, tracking_frame):
    """
    Combine physics features with learned embeddings.
    """
    physics = extract_physics_features(game_state)

    # Add to graph node features
    graph = game_state_to_graph(tracking_frame, game_state.ball_position)

    # Append physics features as graph-level features
    graph.physics_features = torch.tensor([
        physics['elimination_ratio'],
        physics['position_score'],
        physics['density_score']
    ])

    return graph
```

---

## Part 6: Tracking Pipeline Enhancement

### Current Pipeline (from codebase)

Your code uses:
- YOLOv8 for detection
- ByteTrack for tracking
- Homography for coordinate transformation

### Enhancements Needed

```python
# 1. Improve team classification
# Current: color clustering
# Enhancement: jersey number detection or better color model

from ultralytics import YOLO

# Fine-tune YOLO on football-specific data
model = YOLO('yolov8n.pt')
model.train(
    data='path/to/football_dataset.yaml',
    epochs=100,
    imgsz=1280
)

# 2. Add re-identification for occlusion handling
# Use a ReID model to match players across occlusions

from torchreid import models

reid_model = models.build_model(
    name='osnet_x1_0',
    num_classes=23,  # max players + ref
    pretrained=True
)

def extract_appearance(crop):
    """Extract appearance embedding for ReID."""
    return reid_model(crop)

def match_across_occlusion(lost_track, candidates):
    """Match lost track to candidate detections."""
    lost_embedding = extract_appearance(lost_track.last_crop)
    best_match = None
    best_score = 0

    for candidate in candidates:
        cand_embedding = extract_appearance(candidate.crop)
        score = cosine_similarity(lost_embedding, cand_embedding)
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score > 0.7 else None
```

---

## Part 7: Game Model Alignment System

### Define Game Model Principles

```python
from dataclasses import dataclass
from enum import Enum

class PrincipleType(Enum):
    BUILDUP = "buildup"
    COUNTER_PRESS = "counter_press"
    TRANSITION = "transition"
    FINAL_THIRD = "final_third"

@dataclass
class GameModelPrinciple:
    name: str
    type: PrincipleType
    trigger_condition: callable
    execution_criteria: callable

# Example: Counter-press principle
counter_press = GameModelPrinciple(
    name="Counter Press on Loss",
    type=PrincipleType.COUNTER_PRESS,
    trigger_condition=lambda state: (
        state.possession_changed and
        state.zone in ['HIGH_LOSS', 'MID_LOSS']
    ),
    execution_criteria=lambda state: (
        state.nearest_defender_to_ball_distance < 5.0 and
        state.pressing_players_count >= 2 and
        state.time_since_loss < 3.0
    )
)
```

### Measure Execution

```python
def evaluate_principle_execution(match_data, principle):
    """
    Evaluate how well a principle was executed across a match.
    """
    trigger_moments = []
    execution_scores = []

    for frame in match_data.frames:
        state = extract_game_state(frame)

        if principle.trigger_condition(state):
            trigger_moments.append(frame)
            executed = principle.execution_criteria(state)
            execution_scores.append(1.0 if executed else 0.0)

    return {
        'principle': principle.name,
        'trigger_count': len(trigger_moments),
        'execution_rate': sum(execution_scores) / len(execution_scores),
        'moments': trigger_moments
    }
```

---

## Part 8: Counterfactual Analysis Engine

### Generate Alternative Actions

```python
def generate_counterfactuals(game_state, player):
    """
    Generate all reasonable alternative actions for a player.
    """
    alternatives = []

    # Pass options
    for teammate in game_state.teammates:
        if can_pass_to(player, teammate, game_state.defenders):
            alternatives.append({
                'type': 'pass',
                'target': teammate,
                'success_prob': estimate_pass_success(player, teammate, game_state),
                'expected_value': evaluate_after_pass(game_state, player, teammate)
            })

    # Dribble options
    for direction in [0, 45, 90, 135, 180, 225, 270, 315]:
        if can_dribble(player, direction, game_state.defenders):
            alternatives.append({
                'type': 'dribble',
                'direction': direction,
                'success_prob': estimate_dribble_success(player, direction, game_state),
                'expected_value': evaluate_after_dribble(game_state, player, direction)
            })

    # Shot option
    if in_shooting_range(player):
        alternatives.append({
            'type': 'shot',
            'target': 'goal',
            'success_prob': calculate_xg(player, game_state),
            'expected_value': calculate_xg(player, game_state)  # xG is EV for shots
        })

    return sorted(alternatives, key=lambda x: x['expected_value'], reverse=True)
```

### Compare Actual vs Optimal

```python
def analyze_decision_quality(actual_action, game_state, player):
    """
    Analyze whether the player made the optimal decision.
    """
    alternatives = generate_counterfactuals(game_state, player)

    # Find the actual action in alternatives
    actual_ev = None
    for alt in alternatives:
        if matches_action(alt, actual_action):
            actual_ev = alt['expected_value']
            break

    optimal = alternatives[0]
    optimal_ev = optimal['expected_value']

    # Calculate decision quality
    if actual_ev is None:
        # Action not in our model
        return {'quality': 'unknown', 'optimal': optimal}

    ev_gap = optimal_ev - actual_ev

    if ev_gap < 0.02:
        quality = 'optimal'
    elif ev_gap < 0.05:
        quality = 'good'
    elif ev_gap < 0.10:
        quality = 'suboptimal'
    else:
        quality = 'poor'

    return {
        'quality': quality,
        'actual_ev': actual_ev,
        'optimal_ev': optimal_ev,
        'ev_gap': ev_gap,
        'optimal_action': optimal,
        'all_alternatives': alternatives
    }
```

---

## Part 9: Visualization Layer

### Pitch Visualization with mplsoccer

```python
from mplsoccer import Pitch
import matplotlib.pyplot as plt

def visualize_decision_moment(game_state, alternatives, actual_action):
    """
    Visualize a decision moment with alternatives.
    """
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))

    # Plot players
    for player in game_state.teammates:
        pitch.scatter(player.x, player.y, ax=ax, c='blue', s=200)

    for player in game_state.defenders:
        pitch.scatter(player.x, player.y, ax=ax, c='red', s=200)

    # Plot ball carrier
    carrier = game_state.ball_carrier
    pitch.scatter(carrier.x, carrier.y, ax=ax, c='yellow', s=300, marker='*')

    # Plot alternatives with line thickness = expected value
    for alt in alternatives[:5]:  # Top 5 options
        if alt['type'] == 'pass':
            target = alt['target']
            pitch.lines(
                carrier.x, carrier.y, target.x, target.y,
                ax=ax, lw=alt['expected_value'] * 10,
                color='green' if alt == alternatives[0] else 'gray',
                alpha=0.6
            )

    # Highlight actual action
    if actual_action['type'] == 'pass':
        target = actual_action['target']
        pitch.lines(
            carrier.x, carrier.y, target.x, target.y,
            ax=ax, lw=3, color='orange', linestyle='--'
        )

    plt.title(f"Decision Quality: {analyze_decision_quality(actual_action, game_state, carrier)['quality']}")
    return fig
```

---

## Part 10: Development Timeline

### Month 1-2: Foundation

**Week 1-2:**
- [ ] Set up development environment
- [ ] Install all libraries (kloppy, socceraction, mplsoccer, torch)
- [ ] Load StatsBomb open data successfully
- [ ] Run socceraction VAEP tutorial end-to-end

**Week 3-4:**
- [ ] Implement VAEP on Marshall's available data
- [ ] Generate first player action value report
- [ ] Get feedback from coaching staff

**Week 5-6:**
- [ ] Connect existing tracking pipeline to kloppy format
- [ ] Validate tracking accuracy on sample video
- [ ] Document coordinate system and transformations

**Week 7-8:**
- [ ] Define Marshall game model principles in code
- [ ] Build moment classifier for 3 key principles
- [ ] Generate first game model alignment report

### Month 3-4: Core Capabilities

**Week 9-10:**
- [ ] Implement counterfactual action generator
- [ ] Build decision quality scoring
- [ ] Create visualization for decision moments

**Week 11-12:**
- [ ] Build GNN model for outcome prediction
- [ ] Train on StatsBomb data
- [ ] Validate prediction accuracy

**Week 13-14:**
- [ ] Integrate physics features with GNN
- [ ] Experiment with different architectures
- [ ] Document what works, what doesn't

**Week 15-16:**
- [ ] Build set piece analyzer (corners focus)
- [ ] Generate positioning recommendations
- [ ] Validate with coaching staff

### Month 5-6: Integration and Validation

**Week 17-18:**
- [ ] Integrate all components into unified pipeline
- [ ] Build API for accessing analysis
- [ ] Create coaching dashboard MVP

**Week 19-20:**
- [ ] Process 10+ Marshall matches through full pipeline
- [ ] Generate comprehensive analysis reports
- [ ] Gather systematic feedback

**Week 21-22:**
- [ ] Iterate based on feedback
- [ ] Fix accuracy issues
- [ ] Improve UX

**Week 23-24:**
- [ ] Prepare for second customer
- [ ] Document system for external use
- [ ] Create pitch materials

---

## Part 11: Success Metrics

### Technical Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Tracking position error | < 2m | Compare to GPS ground truth |
| VAEP calibration | ECE < 0.05 | Expected calibration error |
| Decision quality model accuracy | > 70% | Does top-ranked action match coach assessment? |
| Set piece model | 80% coach preference | A/B test vs real tactics |

### Business Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Coach satisfaction | > 4/5 | Feedback survey |
| Time saved | 5+ hours/week | Coach self-report |
| Adoption | Used for every match | Usage tracking |
| Second customer | 1 paying customer | Contract signed |

---

## Appendix: Open Source Resources

### Datasets
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Metrica Sample Data](https://github.com/metrica-sports/sample-data)
- [Wyscout Public Dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000)

### Libraries
- [kloppy](https://github.com/PySport/kloppy)
- [socceraction](https://github.com/ML-KULeuven/socceraction)
- [mplsoccer](https://github.com/andrewRowlinson/mplsoccer)
- [Football-Players-Tracking](https://github.com/Darkmyter/Football-Players-Tracking)

### Tutorials
- [Friends of Tracking YouTube](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w)
- [VAEP Tutorial Notebooks](https://github.com/soccer-analytics-research/fot-valuing-actions)
- [ECML 2024 Sports Analytics Tutorial](https://dtai.cs.kuleuven.be/tutorials/sports/)

### Papers
- [TacticAI (DeepMind, 2024)](https://www.nature.com/articles/s41467-024-45965-x)
- [VAEP (Decroos et al., 2019)](https://arxiv.org/abs/1802.07127)
- [EPV Revisited (2025)](https://arxiv.org/abs/2502.02565)
