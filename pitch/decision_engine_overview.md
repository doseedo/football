# Decision Engine
## AI-Powered Tactical Intelligence for Player Development

---

## What Is the Decision Engine?

The Decision Engine is an AI system that goes beyond tracking *where* players are — it understands *what's happening tactically* and *what should happen next*.

**Three Core Capabilities:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION ENGINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. TACTICAL STATE        2. SPATIAL           3. MOVEMENT │
│      RECOGNITION              ANALYSIS             PREDICTION│
│                                                             │
│   "What phase of play?"   "Who controls       "Where should │
│   "Who has possession?"    what space?"        players be?" │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Tactical State Recognition

### What It Does
Classifies every moment of the match into tactical phases:

| State | Description | Use Case |
|-------|-------------|----------|
| **ATTACKING** | In possession, building or finishing | Track attacking patterns |
| **DEFENDING** | Out of possession, organized defense | Analyze defensive shape |
| **TRANSITION_ATTACK** | Won ball, counterattacking | Measure counter speed |
| **TRANSITION_DEFENSE** | Lost ball, recovering shape | Identify vulnerability windows |
| **SET_PIECE** | Dead ball situations | Set piece analysis |

### How It Works
A Graph Neural Network (GNN) analyzes the spatial relationships between all 22 players and the ball:

```
Input: Player positions, velocities, ball position
   ↓
Graph Construction: Each player = node, relationships = edges
   ↓
GNN Processing: Learn team coordination patterns
   ↓
Classification: Output tactical state + confidence score
```

### Player Development Application

**Team Level:**
- "We spent 62% of the match defending — why?"
- "Our transition to attack takes 4.2 seconds on average — opponent takes 3.1 seconds"
- "We lose defensive shape within 2 seconds of losing possession"

**Individual Level:**
- "Player #6 is in correct defensive position 78% of the time during DEFENDING state"
- "Player #10's recovery runs during TRANSITION_DEFENSE are 0.8 seconds slower than target"

---

## 2. Spatial Analysis

### Metrics Calculated

#### Team Structure
| Metric | What It Measures |
|--------|------------------|
| Defensive Line Height | How high/deep the back line plays (meters from goal) |
| Offensive Line Height | Forward positioning of front players |
| Team Length | Vertical compactness (defensive to offensive line) |
| Team Width | Horizontal spread across the pitch |
| Team Compactness | Average distance between teammates |

#### Space Control
| Metric | What It Measures |
|--------|------------------|
| Pitch Control % | What percentage of the pitch each team "owns" |
| Voronoi Areas | Space each individual player controls |
| Passing Lane Count | Number of viable passing options |
| Passing Lane Quality | How open/contested those lanes are |

#### Pressure & Threat
| Metric | What It Measures |
|--------|------------------|
| Pressure on Ball | Defensive pressure intensity (0-1 scale) |
| Expected Threat (xT) | Danger level of current ball position |
| Threatening Players | Attackers in dangerous positions |

### How It Works

**Voronoi Tessellation:**
The pitch is divided into zones based on which player can reach each point fastest:

```
    ┌────────────────────────────────────┐
    │    ╱╲      ╱╲      ╱╲      ╱╲      │
    │   ╱  ╲    ╱  ╲    ╱  ╲    ╱  ╲     │
    │  ╱ P1 ╲  ╱ P2 ╲  ╱ P3 ╲  ╱ P4 ╲    │
    │ ╱      ╲╱      ╲╱      ╲╱      ╲   │
    │╱        ╲      ╱╲      ╱        ╲  │
    │╲        ╱╲    ╱  ╲    ╱╲        ╱  │
    │ ╲  P5  ╱  ╲  ╱ P6 ╲  ╱  ╲  P7  ╱   │
    │  ╲    ╱    ╲╱      ╲╱    ╲    ╱    │
    │   ╲  ╱              ╲╱    ╲  ╱     │
    └────────────────────────────────────┘

    Each zone = space that player controls
    Sum of team zones = pitch control %
```

### Player Development Application

**Defensive Positioning:**
> "When we're in a mid-block, our defensive line should be at 40m. In the loss to [opponent], it dropped to 32m in the second half — that's why they dominated possession in our half."

**Space Creation:**
> "Player #11 creates 15% more space than the team average when making runs — that's why they get open. Player #9 needs to work on timing runs to create separation."

**Pressing Coordination:**
> "Our pressing intensity drops from 0.74 to 0.58 after 70 minutes. Conditioning focus for the #6 and #8 positions."

---

## 3. Movement Prediction

### What It Does
Predicts where players *should* move based on the tactical situation — then compares to where they *actually* go.

### The Model: Baller2Vec++

A transformer-based neural network trained on professional match data:

```
Input (Current Frame):
├── All 22 player positions
├── All 22 player velocities
├── Ball position and velocity
├── Tactical state
└── Team coordination patterns

   ↓ Transformer Processing ↓

Output (Next 1-3 seconds):
├── Predicted position for each player
├── Confidence interval
├── Optimal position given tactical context
└── Deviation from optimal (coaching feedback)
```

### Key Features

**Team Coordination Modeling:**
The model understands that players don't move independently — it learns:
- Defensive line moves as a unit
- Pressing triggers coordinate multiple players
- Attacking runs create space for teammates

**Physics Constraints:**
Predictions respect human physical limits:
- Maximum acceleration/deceleration
- Realistic top speeds
- Pitch boundaries

### Player Development Application

**Positional Coaching:**
```
FRAME ANALYSIS — 23:41

Player #6 (Defensive Mid)
├── Actual Position: (34.2, 28.1)
├── Optimal Position: (38.5, 31.2)
├── Deviation: 5.2m
└── Coaching Note: "Should be 5m higher and wider
                    to cut passing lane to #10"

[Visualization shows ghost position where player should be]
```

**Pattern Recognition:**
> "Player #4 consistently undercommits on press triggers — average 2.1m behind optimal starting position. This delays team press by 0.4 seconds."

**Scenario Simulation:**
> "If we press with 4 players instead of 3 in this situation, the model predicts we win the ball 23% more often but leave 12% more space behind."

---

## Decision Engine → Player Development Workflow

### Step 1: Baseline Assessment

Run 3-5 matches through the system to establish player baselines:

```
PLAYER BASELINE PROFILE

Physical:
├── Avg Distance: 9,847m/match
├── Sprint Frequency: 32/match
├── High-Intensity Minutes: 18.4
└── Fatigue Point: 68th minute

Positional:
├── Correct Position Rate: 74%
├── Avg Deviation from Optimal: 4.1m
├── Recovery Run Speed: 6.2 m/s
└── Pressing Trigger Response: 1.2s

Tactical:
├── Space Created: +8% vs team avg
├── Passing Lane Blocking: 67%
└── Transition Involvement: 3.2 actions/transition
```

### Step 2: Identify Development Areas

Compare player profiles to:
- Team tactical requirements
- Position-specific benchmarks
- Top performers at same position

```
DEVELOPMENT PRIORITY MATRIX

                    Impact
                    High    │    Low
              ┌─────────────┼─────────────┐
         High │ PRIORITY 1  │ PRIORITY 3  │
    Gap       │ Fix Now     │ Monitor     │
              ├─────────────┼─────────────┤
         Low  │ PRIORITY 2  │ IGNORE      │
              │ Optimize    │             │
              └─────────────┴─────────────┘

Player #8 Development Priorities:
1. Pressing trigger response (1.2s → target 0.8s) — HIGH IMPACT, HIGH GAP
2. Fatigue point (68min → target 75min) — HIGH IMPACT, MEDIUM GAP
3. Sprint frequency (32 → target 36) — MEDIUM IMPACT, LOW GAP
```

### Step 3: Training Prescription

Use simulation to design targeted training:

**Example: Improving Pressing Triggers**

1. **Identify Problem:** Video clips where player was slow to press
2. **Show Optimal:** Overlay showing where they should have been
3. **Drill Design:** Small-sided game emphasizing trigger recognition
4. **Track Progress:** Re-measure after 2 weeks of focused training

### Step 4: Progress Monitoring

Weekly/monthly tracking of development metrics:

```
PLAYER #8 — DEVELOPMENT TRACKER

Metric              | Baseline | Week 4 | Week 8 | Target
--------------------|----------|--------|--------|--------
Pressing Response   | 1.2s     | 1.0s   | 0.85s  | 0.8s  ▲
Optimal Position %  | 74%      | 77%    | 81%    | 85%   ▲
Fatigue Point       | 68min    | 70min  | 72min  | 75min ▲
Sprint Count        | 32       | 33     | 34     | 36    →
```

### Step 5: Match Day Application

Pre-match: Review opponent tendencies
- "They press high when GK has ball — look for outlet to #7"
- "Their #6 is slow to recover — target that channel in transition"

In-match: Real-time physical monitoring (if processing live)
- "Player #10 physical output dropping — substitute consideration"

Post-match: Performance vs. plan
- "We executed high press 78% of opportunities — target was 85%"
- "Transition speed improved from 4.1s to 3.6s average"

---

## Recruiting Application

### Prospect Evaluation Framework

Process recruit's club/high school footage and compare to Marshall profiles:

```
RECRUIT EVALUATION: [Name], CM, Class of 2026

PHYSICAL FIT                          TACTICAL FIT
├── Distance: 10,234m (▲ Marshall avg) ├── Pressing Style: HIGH (matches)
├── Sprint Profile: Similar to #8      ├── Position Discipline: 82% (▲ avg)
├── Top Speed: 30.1 km/h (meets min)   ├── Transition Speed: FAST
└── Fatigue: Late (75min+)             └── Space Creation: +18% (excellent)

PROJECTION
├── Best Comparable: Current #8 (Year 2 level)
├── Development Areas: Defensive recovery timing
├── System Fit Score: 87/100
└── RECOMMENDATION: Priority recruit — fits system well
```

### Competitive Advantage
> "We evaluated you using the same AI system professional clubs use. Here's your performance profile compared to our current midfielders. You'd fit right in at this position, and here's the development path we see for you."

No other Sun Belt program can offer this level of analytical sophistication in recruiting.

---

## Summary: Decision Engine Value

| Capability | Traditional Approach | With Decision Engine |
|------------|---------------------|----------------------|
| Tactical analysis | Watch film, take notes | Quantified metrics, automated |
| Player positioning | "Feel" for who's out of position | Exact deviation in meters |
| Development tracking | Subjective coach assessment | Objective trend data |
| Opponent scouting | Manual tagging, hours of work | Automated pattern recognition |
| Recruiting evaluation | Eye test, limited data | Systematic comparison to roster |
| Training design | Experience-based | Data-driven prescription |

**Bottom Line:** The Decision Engine transforms coaching intuition into measurable, trackable, improvable data — while still leaving all decisions to the coaching staff.
