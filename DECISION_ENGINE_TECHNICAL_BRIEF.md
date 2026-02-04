# Decision Engine - Technical Brief

This document provides technical details on the decision engine architecture for those who want to understand how it works under the hood.

---

## Overview

The decision engine is software that takes player coordinates and produces tactical analysis. It sits at the end of the tracking pipeline:

```
Video → Detection → Tracking → Coordinates → Decision Engine → Analysis
        (YOLO)    (ByteTrack) (Calibration)   (Elimination,    (Insights)
                                               Forces, Scoring)
```

**Input:** Player and ball coordinates (x, y) for each frame
**Output:** Tactical measurements, scores, and flagged moments

---

## Core Modules

```
src/decision_engine/
├── __init__.py           # Public API exports
├── pitch_geometry.py     # Coordinate system and spatial utilities
├── elimination.py        # Core elimination logic (attacking metric)
├── defense_physics.py    # Attraction-based defensive behavior
├── state_scoring.py      # Game state evaluation system
├── block_models.py       # Defensive block configurations
└── visualizer.py         # Tactical board visualization
```

---

## Module 1: Elimination Calculator

### Concept

A defender is **eliminated** when:
1. The ball is past them (positionally, toward goal)
2. They cannot reach an effective intervention point before the attacker can progress to a more dangerous position

This is binary at the moment of evaluation. A defender is either eliminated or not.

**Key insight:** A defender who is goal-side but functionally irrelevant (cannot intervene in time) is still eliminated.

### How It Works

For each defender, the calculator:

1. **Determines if goal-side** — Is the defender between ball and goal?

2. **Finds intervention point** — The closest point on the ball-to-goal line that the defender could reach

3. **Calculates defender time** — Time for defender to reach intervention point, accounting for:
   - Current position
   - Current velocity/momentum
   - Reaction time (default: 0.25s)
   - Max speed (default: 8 m/s)

4. **Calculates attacker time** — Time for ball carrier to reach or pass through the intervention point

5. **Compares times** — If defender time > attacker time → ELIMINATED

### Output

```python
EliminationState:
    ball_position: Position
    ball_carrier: Player
    defenders: List[EliminationResult]

    # Derived
    eliminated_count: int
    active_count: int
    elimination_ratio: float  # 0 to 1
```

### Why It Matters

Every attacking action can be evaluated by how many defenders it eliminates. A pass that takes out three defenders is more valuable than one that takes out one. This quantifies attacking progress objectively.

---

## Module 2: Defensive Force Model

### Concept

Defensive positioning is modeled as an equilibrium of attraction forces. Each defender experiences pulls toward multiple targets:

| Force | Effect | Description |
|-------|--------|-------------|
| Ball attraction | Pressing | Pull toward ball position |
| Goal attraction | Protection | Pull toward own goal |
| Zone attraction | Structure | Pull toward assigned area |
| Opponent attraction | Marking | Pull toward dangerous attackers |
| Teammate repulsion | Spacing | Push away from nearby defenders |
| Line attraction | Compactness | Pull to maintain line height |
| xG path blocking | Lane coverage | Pull into highest-value passing lanes |

The equilibrium of these forces determines **optimal positioning**.

### Adjustable Weights

The key insight: **force weights are tunable parameters**.

```python
DefensiveForceModel(
    ball_weight=1.0,      # How much we press
    goal_weight=0.6,      # How deep we sit
    zone_weight=0.4,      # How zonal vs reactive
    opponent_weight=0.8,  # How tight we mark
    teammate_repulsion=0.3,
    line_weight=0.5,      # How compact we stay
)
```

Different teams defend differently:
- High ball weight → aggressive pressing team
- High goal weight → deep-sitting team
- High opponent weight → tight man-markers
- High line weight → compact block

### Two Applications

**1. Modeling opponents**
Watch their film, adjust weights until the model matches their actual positioning patterns. Now we have a mathematical representation of how they defend — and can simulate where they'll be in different situations.

**2. Measuring ourselves**
Define our target weights (how we want to defend). Measure actual matches against those targets. The gap between model and reality shows exactly where and when we break from our principles.

### xG Path Calculation

The model calculates the ball's highest xG path to goal at any moment — the sequence of passes or carries that would create the most dangerous chance.

Good defensive positioning blocks these paths. Defenders can look like they're in position but actually leave the most dangerous route unguarded. The model detects this.

---

## Module 3: Game State Evaluator

### Concept

Every game state receives a composite score based on multiple factors. Higher score = more advantageous for attacking team.

### Components

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Elimination | 25% | Defenders functionally out of play |
| Proximity | 20% | Distance to goal |
| Angle | 15% | Shooting angle available |
| Density | 15% | Space around the ball |
| Compactness | 10% | Defensive structure integrity |
| Action | 15% | Forward passing/dribbling options |

### Output

```python
StateScore:
    elimination_score: float    # 0-1
    proximity_score: float      # 0-1
    angle_score: float          # 0-1
    density_score: float        # 0-1
    compactness_score: float    # 0-1
    action_score: float         # 0-1

    total: float  # Weighted combination
```

### Why It Matters

This creates an objective score for any position. We can:
- Compare different moments in a match
- Rank ball locations by value
- Identify why certain attacks succeeded while others stalled
- Track trends across a season

---

## Module 4: Defensive Block Models

### Concept

Three primary block configurations with different characteristics:

| Block | Line Height | Characteristics |
|-------|-------------|-----------------|
| **Low** | ~12m from goal | Compact, protective, minimal pressing |
| **Mid** | ~22m from goal | Balanced, controls midfield space |
| **High** | Near halfway | Aggressive, traps opponents in their half |

### What It Measures

For any moment, the model can determine:
- Which block we're in (or transitioning between)
- Whether we're maintaining the intended block
- Where gaps exist in the block structure
- What triggered a block breakdown

---

## Module 5: Visualizer

Renders analysis on a tactical board using matplotlib/mplsoccer:

- Game state visualization (players, ball, eliminations)
- Value heatmaps (where is space valuable?)
- Force diagrams (what's pulling defenders where?)
- Block shape overlays

---

## Use Cases

### 1. Post-Match Analysis

Feed match coordinates into the engine. For any moment:
- How many defenders were eliminated?
- Were we in our intended shape?
- What was the gap between actual and optimal positioning?
- Which specific players were out of position?

Instead of watching and subjectively noting "we looked stretched," we have: "Lines were 28m apart (target: 20m), 3 defenders eliminated, breakdown triggered at 23:41 when #6 drifted central."

### 2. What-If Simulation

After a goal conceded:
1. Engine shows the moment of breakdown
2. Identifies which defenders were eliminated, where structure failed
3. Simulates alternatives: "If Player X had been 3m deeper, would they still have been eliminated?"
4. Quantifies the difference

This turns review into actionable insights — not just identifying what went wrong, but testing what would have fixed it.

### 3. Opponent Modeling

1. Feed opponent match video through tracking
2. Extract their positioning data
3. Tune the defensive force model to match their patterns
4. Now have a mathematical representation of their tendencies
5. Simulate scenarios: "When we overload the right, where do they become vulnerable?"

### 4. Training Design

Use the engine in simulation mode (no video needed):
- Place players on a tactical board
- Engine calculates eliminations, optimal positions, game state scores
- Test different shapes and scenarios computationally
- Design exercises around the findings

---

## Philosophy: Structure Before Talent

The base model treats all players as physically equal:
- Same max speed (8 m/s)
- Same reaction time (0.25s)
- No fatigue, psychology, or randomness

This is intentional. **We understand structure before we layer in talent.**

If positioning fails when everyone's equal, it will fail when talent is added. Talent is a modifier on top of sound structure, not a replacement for it.

This also identifies whether breakdowns are:
- **Positional** (fixable through coaching)
- **Physical** (need different personnel)

---

## Technical Assumptions

| Assumption | Implication |
|------------|-------------|
| Equal player speed | No talent-based advantage in base model |
| Instant acceleration | Simplified physics |
| Constant pass speed | No curve, no weighted balls |
| 2D only | No aerial play |
| Static goalkeeper | Not included in elimination calcs |

These are intentional simplifications. Layers can be added as the system matures.

---

## Dependencies

```
numpy
matplotlib
mplsoccer (optional, for enhanced pitch rendering)
```

---

## Current State

The modules exist and work in isolation. What remains:

1. **Tracking validation** — Engine output is only as good as coordinate input
2. **Parameter tuning** — Force weights need calibration with coaching staff
3. **Face validity** — Confirm engine flags the same moments coaches identify
4. **Integration** — Connect to Part 1 (Hub) for queryable insights

---

## Example Usage

```python
from decision_engine import (
    Position,
    Player,
    GameState,
    GameStateEvaluator,
    EliminationCalculator,
    DefensiveForceModel,
)

# Create game state
state = GameState(
    ball_position=Position(25, 0),
    ball_carrier=attacker,
    attackers=[a1, a2, a3],
    defenders=[d1, d2, d3, d4],
)

# Evaluate
evaluator = GameStateEvaluator()
result = evaluator.evaluate(state)

print(f"Score: {result.score.total:.2f}")
print(f"Eliminated: {result.elimination_state.eliminated_count}")

# Model defensive positioning
force_model = DefensiveForceModel(ball_weight=1.2, goal_weight=0.5)
ideal_positions = force_model.calculate_equilibrium(defenders, ball_position)

# Compare actual vs ideal
for defender, ideal in zip(defenders, ideal_positions):
    gap = defender.position.distance_to(ideal)
    print(f"{defender.id}: {gap:.1f}m from optimal")
```

---

*This document accompanies MARSHALL_SOCCER_AI_PITCH.md*
*Last updated: February 2026*
