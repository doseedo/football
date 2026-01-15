# Football Decision Engine - Improvement Roadmap

## Current State Analysis

Your system is a **tactical football analysis engine** that treats football positions like a turn-based strategy game with:
- **Elimination detection** - determining which defenders are "out of the play"
- **Physics-based defense modeling** - attraction forces guiding defensive positions
- **State scoring** - composite evaluation of game states (6 components)
- **Action evaluation** - ranking shots, passes, dribbles by expected value
- **Block configurations** - low/mid/high defensive setups

**Strengths:**
- Well-architected modular design
- Solid physics/geometry foundation
- Good visualization with mplsoccer
- Clean API with comprehensive `__init__.py` exports

**Gaps Identified:**
- No unit tests for decision engine
- Static analysis only (no temporal reasoning)
- No learning/training capability
- No integration with actual match data
- Action probabilities use simplified heuristics

---

## Recommended Improvements (Priority Order)

### 1. Add Comprehensive Testing
**Priority: HIGH**
**Impact: Foundation for all future work**

The decision engine has **zero test coverage**. Before adding features, establish a test suite.

```
tests/
├── decision_engine/
│   ├── test_elimination.py       # Elimination edge cases
│   ├── test_state_scoring.py     # Score calculation verification
│   ├── test_defense_physics.py   # Force calculations
│   ├── test_block_models.py      # Block position accuracy
│   └── test_integration.py       # Full pipeline tests
```

**Key test scenarios:**
- Boundary cases: ball at goal line, corners, midfield
- Edge cases: no defenders, all defenders eliminated, crowded penalty area
- Regression tests: known scenarios with expected scores

### 2. Temporal Reasoning (Multi-Move Lookahead)
**Priority: HIGH**
**Impact: Strategic depth**

Currently, state evaluation is **single-frame only**. Add lookahead:

```python
class TemporalEvaluator:
    """Evaluate sequences of states, not just snapshots."""

    def evaluate_sequence(self, states: List[GameState], depth: int = 3) -> float:
        """
        Minimax-style evaluation:
        - Attackers maximize score
        - Defenders minimize score
        - Alpha-beta pruning for efficiency
        """
        pass

    def best_action_sequence(self, state: GameState, horizon: int = 5) -> List[ActionOption]:
        """Find optimal action sequence using Monte Carlo Tree Search."""
        pass
```

**Key additions:**
- Player movement prediction (use existing `src/extrapolation/`)
- Opponent response modeling
- Pass/dribble/shot sequencing

### 3. Data-Driven Action Probabilities
**Priority: HIGH**
**Impact: Accuracy**

Current success probabilities are heuristic (e.g., `dist_factor * angle_factor`). Train on real data.

**Approach:**
1. Collect labeled outcome data from matches:
   - Shot attempts → goal/save/miss/block
   - Passes → completed/intercepted
   - Dribbles → successful/dispossessed

2. Train classifiers:
```python
class LearnedActionModel:
    """Data-driven action success prediction."""

    def __init__(self):
        self.shot_model = XGBClassifier()  # Or neural network
        self.pass_model = XGBClassifier()
        self.dribble_model = XGBClassifier()

    def predict_shot_success(self, features: ShotFeatures) -> float:
        """Features: distance, angle, defenders, GK position, etc."""
        return self.shot_model.predict_proba(features)[1]
```

3. Integrate with SoccerNet or StatsBomb data

### 4. Goalkeeper Modeling
**Priority: MEDIUM**
**Impact: Shot evaluation accuracy**

The current system doesn't model the goalkeeper as a special entity.

```python
@dataclass
class Goalkeeper(Player):
    """Specialized goalkeeper with diving/positioning attributes."""
    reach: float = 2.0           # Diving reach
    reaction_time: float = 0.15  # Seconds
    dive_speed: float = 4.0      # m/s lateral

class GKShotModel:
    """Model goalkeeper's ability to save shots."""

    def save_probability(self, shot: ShotAttempt, gk: Goalkeeper) -> float:
        """
        Consider:
        - Shot speed and placement
        - GK starting position
        - Reaction time + dive trajectory
        - Ball trajectory (high/low, driven/chipped)
        """
        pass
```

### 5. Expected Goals (xG) Integration
**Priority: MEDIUM**
**Impact: Industry-standard metrics**

Replace custom shot scoring with industry-standard xG.

```python
class XGModel:
    """Expected Goals model for shot evaluation."""

    def calculate_xg(self, shot: ShotContext) -> float:
        """
        Standard xG features:
        - Distance to goal
        - Angle to goal
        - Shot body part (foot/head)
        - Assist type (through ball, cross, cutback)
        - Defensive pressure
        - GK position
        """
        pass
```

Consider using or training on:
- StatsBomb open xG data
- Understat public xG models
- Custom model trained on SoccerNet

### 6. Formation Recognition & Analysis
**Priority: MEDIUM**
**Impact: Tactical intelligence**

Automatically detect formations from player positions.

```python
class FormationDetector:
    """Detect team formations from positions."""

    FORMATIONS = ["4-4-2", "4-3-3", "3-5-2", "4-2-3-1", ...]

    def detect(self, players: List[Player]) -> str:
        """Classify formation using clustering or neural network."""
        pass

    def formation_vulnerability(self, formation: str, block_type: BlockType) -> Dict:
        """Identify weak zones in formation against specific blocks."""
        pass
```

### 7. Passing Network Analysis
**Priority: MEDIUM**
**Impact: Team coordination insights**

Leverage existing GNN infrastructure for passing analysis.

```python
class PassingNetwork:
    """Analyze team passing patterns and combinations."""

    def build_network(self, passes: List[PassEvent]) -> nx.DiGraph:
        """Create directed graph of passing relationships."""
        pass

    def identify_combinations(self) -> List[PassingTriangle]:
        """Find common triangular passing combinations."""
        pass

    def key_player_centrality(self) -> Dict[str, float]:
        """Identify playmakers via betweenness centrality."""
        pass
```

### 8. Real-Time Integration Pipeline
**Priority: MEDIUM**
**Impact: Live analysis capability**

Connect decision engine to video processing pipeline.

```python
class LiveTacticalAnalysis:
    """Real-time tactical analysis from video stream."""

    def __init__(self, video_source: str):
        self.detector = PlayerDetector()
        self.tracker = PlayerTracker()
        self.homography = HomographyEstimator()
        self.evaluator = GameStateEvaluator()

    async def process_frame(self, frame) -> AnalyzedState:
        """Pipeline: detect → track → transform → evaluate."""
        pass

    def stream_analysis(self) -> AsyncIterator[AnalyzedState]:
        """Yield analyzed states in real-time."""
        pass
```

### 9. Set Piece Analysis Module
**Priority: LOW**
**Impact: Specialized scenarios**

Corners, free kicks, and penalties have different dynamics.

```python
class SetPieceAnalyzer:
    """Analyze set piece situations."""

    def analyze_corner(self, state: GameState) -> CornerAnalysis:
        """
        Evaluate:
        - Delivery zones (near/far post, edge of box)
        - Marking assignments
        - Space at back post
        - Counter-attack risk
        """
        pass

    def analyze_free_kick(self, state: GameState, position: Position) -> FreeKickAnalysis:
        """Direct shot vs cross vs short pass evaluation."""
        pass
```

### 10. Scenario Simulation Engine
**Priority: LOW**
**Impact: What-if analysis**

Allow manual "what-if" scenario exploration.

```python
class ScenarioSimulator:
    """Simulate tactical scenarios and outcomes."""

    def simulate_action(self, state: GameState, action: ActionOption) -> List[GameState]:
        """Generate possible outcome states from action."""
        pass

    def monte_carlo_evaluation(self, state: GameState, n_simulations: int = 1000) -> float:
        """Monte Carlo simulation of state value."""
        pass

    def optimal_policy(self, state: GameState) -> ActionSequence:
        """Find optimal action sequence via rollouts."""
        pass
```

---

## Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Dependencies |
|---------|----------|--------|--------|--------------|
| Unit Tests | HIGH | Medium | Foundation | None |
| Temporal Lookahead | HIGH | High | Strategic depth | Tests |
| Data-Driven Probabilities | HIGH | High | Accuracy | Training data |
| Goalkeeper Model | MEDIUM | Medium | Shot accuracy | None |
| xG Integration | MEDIUM | Medium | Industry standard | Data |
| Formation Detection | MEDIUM | Medium | Tactical insight | None |
| Passing Networks | MEDIUM | Low | Team analysis | GNN module |
| Live Integration | MEDIUM | High | Real-time use | Pipeline modules |
| Set Pieces | LOW | Medium | Specialized | Base engine |
| Scenario Simulation | LOW | High | What-if analysis | Temporal reasoning |

---

## Quick Wins (Immediate Value)

1. **Add `pytest.ini` and basic tests** - 2-3 test files for core modules
2. **Tune scoring weights** - Current weights are defaults, calibrate with real match data
3. **Add verbose logging** - Use loguru for debugging analysis decisions
4. **Export to common formats** - Add JSON/Parquet export for integration with other tools
5. **CLI tool** - Add typer CLI for command-line analysis

---

## Suggested First Steps

1. Create `tests/decision_engine/` with basic test coverage
2. Run the demo script and verify all visualizations work
3. Collect 10-20 labeled scenarios from real matches for validation
4. Implement goalkeeper modeling (high impact, moderate effort)
5. Add temporal lookahead for 2-3 move sequences

---

## Architecture Recommendation

```
src/decision_engine/
├── __init__.py           # (existing)
├── pitch_geometry.py     # (existing)
├── elimination.py        # (existing)
├── defense_physics.py    # (existing)
├── state_scoring.py      # (existing)
├── block_models.py       # (existing)
├── visualizer.py         # (existing)
├── goalkeeper.py         # NEW: GK-specific modeling
├── temporal.py           # NEW: Multi-move lookahead
├── learned_models.py     # NEW: Data-driven probabilities
├── xg_model.py           # NEW: Expected goals
├── formations.py         # NEW: Formation detection
└── simulation.py         # NEW: Monte Carlo simulation
```

This maintains backward compatibility while adding new capabilities.
