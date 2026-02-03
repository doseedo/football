# Football Decision Engine - Technical Plan

## Vision

**End Goal: A simulation engine grounded in real football data.**

Not a simulation with designed rules that might be wrong. A simulation where the "rules" are learned from thousands of real matches — where we know exactly how fast passes travel, how defenders actually react, what dribbles actually succeed, because we measured it.

The path to get there:

```
Phase 1-4: Learn how football actually works
           ↓
           Collect situations → Extract patterns → Validate accuracy
           ↓
Phase 5:   Build simulation using learned patterns as rules
           ↓
           Agents can explore freely within realistic constraints
           ↓
           Find optimal sequences, novel tactics, better decisions
```

**Why this order:** You can't simulate what you don't understand. First we learn the patterns. Then we simulate with confidence that the simulation matches reality.

---

## The Path: Similarity-Based Learning

### What We're Building First

A **similarity-based learning system** that:
1. Encodes game situations as continuous feature vectors (not discrete categories)
2. Uses similarity algorithms to find matching historical situations
3. Analyzes what actions worked in those similar situations
4. **Learns the actual distributions** — pass success rates, dribble outcomes, defensive responses
5. Builds the knowledge base that will power the simulation

### Key Design Decisions

| Decision | What We Chose | Why |
|----------|---------------|-----|
| Situation matching | Similarity algorithm on continuous features | Hard categories (zones, formation labels) lose nuance and create arbitrary boundaries |
| Defensive encoding | Behavioral features measured from tracking | Formation labels are imprecise - "4-4-2" means many different things |
| Ball position | Continuous coordinates | Zone boundaries are arbitrary - similarity handles "closeness" naturally |
| Skill imbalance | Address in Phase 3 | Build core system first, validate concept, then add complexity |

### Why This Approach

| Alternative | Why Not |
|-------------|---------|
| Pure RL/Simulation | No evidence of real-world tactical transfer (Google Football failed) |
| Board game abstraction | Unvalidated - no proof simplified rules capture real tactics |
| Formation labels | Too imprecise - same label describes very different defensive behaviors |
| Discrete zone categories | Arbitrary boundaries - play 1m apart gets different category |
| Exact position matching | No matches - football state space too large |

**This approach is grounded in:**
- Real tracking data (not simulation)
- Proven methodology (similar to TacticAI retrieval)
- Continuous features that capture actual defensive behavior
- Similarity algorithms that find meaningful matches without arbitrary categorization

---

## Data Requirements

### Primary: Tracking Data
- Full player positions (all 22 players + ball)
- Minimum 25Hz sampling rate
- Source: SkillCorner, Second Spectrum, or equivalent

### Secondary: Event Data
- Pass, shot, dribble, tackle events
- Outcomes (success/failure)
- Source: Wyscout, StatsBomb

### Derived: Context Data
- Score state
- Match time
- Team identities
- Competition level

---

## Situation Encoding

### Ball Position (Continuous)

**NOT discrete zones.** Instead, continuous features:

| Feature | Description | Unit |
|---------|-------------|------|
| `ball_x` | Horizontal position on pitch | Meters from left touchline |
| `ball_y` | Vertical position (depth) | Meters from own goal line |
| `ball_distance_to_goal` | Direct distance to opponent goal center | Meters |
| `ball_angle_to_goal` | Angle to goal from ball position | Degrees |

The similarity algorithm handles "closeness" - situations with ball at 35m and 36m will naturally be similar without needing to define zone boundaries.

### Defensive Features (Behavioral, Not Labels)

**NO formation labels.** Instead, measure actual defensive behavior from tracking data:

#### Core Features (Phase 1)

| Feature | Description | How to Measure |
|---------|-------------|----------------|
| `line_height` | How high/deep is the defensive line | Average Y-coordinate of back 4 (meters from own goal) |
| `compactness_vertical` | Vertical distance between defensive lines | Distance from deepest defender to highest midfielder (meters) |
| `compactness_horizontal` | Width of defensive shape | Distance between widest defenders (meters) |
| `press_intensity` | How aggressively they close down | Average closing speed when ball is received (m/s) |
| `pressure_on_ball` | Immediate pressure on ball carrier | Distance of nearest defender to ball (meters) |
| `players_behind_ball` | Defensive commitment | Count of defenders goal-side of ball |

#### Extended Features (Phase 2 - After Validation)

| Feature | Description | How to Measure |
|---------|-------------|----------------|
| `marking_style` | Man-oriented vs zonal | Correlation coefficient between defender movement and nearest attacker movement over possession |
| `shift_speed` | Defensive reaction time | Average time for defensive shape centroid to adjust after ball movement (seconds) |
| `recovery_tendency` | Counter-press vs drop | Average movement direction (toward/away from ball) in first 2s after possession loss |
| `defensive_width_ratio` | Shape relative to ball | Defensive width divided by distance from ball to goal |

### Attacking Features (Continuous)

| Feature | Description | How to Measure |
|---------|-------------|----------------|
| `attackers_ahead_of_ball` | Offensive presence | Count of attacking players with higher Y than ball |
| `attacking_width` | Spread of attack | Horizontal distance between widest attackers (meters) |
| `players_in_box` | Penalty area threat | Count of attackers in opponent's 18-yard box |
| `central_overload` | Middle congestion | Density of players in central channel |
| `space_ahead` | Room to progress | Nearest defender distance in direction of goal |

### Context Features

| Feature | Description | Values |
|---------|-------------|--------|
| `score_differential` | Current advantage/disadvantage | Integer (..., -2, -1, 0, +1, +2, ...) |
| `match_time` | Minutes played | 0-90+ (continuous) |
| `possession_duration` | Time on ball | Seconds since possession started |

---

## Similarity Matching

### How It Works

1. **Encode each historical situation** as a feature vector containing all features above
2. **Normalize features** to same scale (z-score or min-max)
3. **For a query situation**, compute similarity to all historical situations
4. **Return top-N most similar** situations
5. **Analyze outcomes** of actions taken in those situations

### Distance/Similarity Metric

Weighted Euclidean distance across normalized features:

```
distance(A, B) = sqrt(
    w1 * (A.line_height - B.line_height)² +
    w2 * (A.compactness_vertical - B.compactness_vertical)² +
    w3 * (A.pressure_on_ball - B.pressure_on_ball)² +
    ...
)

similarity = 1 / (1 + distance)
```

**Initial weights:** Equal (1.0 for all features)

**Later:** Learn optimal weights by measuring which weighting best predicts similar outcomes for similar situations

### Why This Works Better Than Categories

| Categories (5x5 zones + formation labels) | Similarity on Continuous Features |
|-------------------------------------------|-----------------------------------|
| Ball at 34m and 36m might be different zones | Ball at 34m and 36m are naturally similar |
| "4-4-2" and "4-4-2 low block" same label | Actual line height distinguishes them |
| Man-to-man and zonal can look identical | Marking correlation feature captures difference |
| Fixed boundaries = arbitrary cutoffs | Similarity is continuous = no arbitrary lines |

---

## Success Metrics

### For Possessions

| Metric | Definition | Use Case |
|--------|------------|----------|
| `xT_gained` | Expected Threat added during possession | Primary success metric |
| `final_third_entry` | Did possession reach attacking third | Progression success |
| `shot_generated` | Did possession result in shot | Chance creation |
| `xG_of_shot` | If shot, what was its xG | Chance quality |

### For Evaluating the System

| Metric | What It Measures |
|--------|------------------|
| Outcome consistency | Do similar situations (high similarity score) have similar outcomes? |
| Coach relevance rating | Do coaches agree retrieved situations are tactically similar? |
| Recommendation accuracy | Do higher-rated actions actually succeed more often? |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                         │
│  ├── Tracking data parser (positions at each frame)         │
│  ├── Event data parser (actions and outcomes)               │
│  └── Data alignment (sync tracking + events by timestamp)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                       │
│  ├── Ball position features (continuous coordinates)        │
│  ├── Defensive behavioral features (from player positions)  │
│  ├── Attacking features (from player positions)             │
│  └── Context features (score, time, etc.)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SITUATION DATABASE                        │
│  ├── Feature vectors for all historical possessions         │
│  ├── Normalized for similarity computation                  │
│  ├── Indexed for fast nearest-neighbor search               │
│  └── Linked to outcomes (what action was taken, did it work)│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SIMILARITY ENGINE                         │
│  ├── Query encoder (same feature extraction)                │
│  ├── Nearest neighbor search (find top-N similar)           │
│  ├── Outcome aggregation (what worked in similar situations)│
│  └── Confidence scoring (based on N and outcome variance)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                           │
│  ├── "In similar situations, X worked Y% of the time"       │
│  ├── Video clips of the most similar historical situations  │
│  ├── Recommended actions ranked by historical success rate  │
│  └── Confidence level based on sample size and consistency  │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Validate Core Concept)

**Goal:** Does similarity matching on behavioral features find meaningful patterns?

**Deliverables:**
- [ ] Tracking data ingestion pipeline
- [ ] Core feature extraction (line height, compactness, pressure, ball position)
- [ ] Situation database with 1000+ possessions
- [ ] Similarity search implementation
- [ ] Basic outcome analysis ("what worked in similar situations")

**Validation Questions:**
- Do situations with high similarity scores have similar outcomes?
- Do coaches agree that retrieved situations are tactically relevant?
- Can we distinguish what works vs what doesn't in similar situations?

**Success Criteria:**
- Coaches rate retrieved situations as relevant >70% of the time
- Outcome variance within similar situations is lower than random baseline

---

### Phase 2: Refinement (Improve Accuracy)

**Goal:** Make similarity matching more accurate, add behavioral features

**Deliverables:**
- [ ] Extended feature set (marking style, shift speed, recovery tendency)
- [ ] Learned feature weights (optimize for outcome prediction)
- [ ] Action-level analysis (not just possession-level)
- [ ] Confidence scores for recommendations
- [ ] Feature importance analysis (which features matter most)

**Validation Questions:**
- Do extended features improve outcome consistency within similar situations?
- Do learned weights outperform equal weights?
- Do coaches prefer recommendations with confidence scores?

---

### Phase 3: Skill Adjustment

**Goal:** Separate "this worked because it's good" from "good players executed it"

**Deliverables:**
- [ ] Player physical metrics integration (speed, acceleration from tracking)
- [ ] Historical performance baselines per player (pass completion rates, etc.)
- [ ] Skill-adjusted success rates
- [ ] Within-team variation analysis (same team, different choices, what worked)

**Validation Questions:**
- Are patterns consistent when controlling for player/team quality?
- Do skill-adjusted recommendations differ from raw recommendations?
- Does skill adjustment improve prediction accuracy?

---

### Phase 4: Production

**Goal:** System ready for regular coaching staff use

**Deliverables:**
- [ ] Fast query interface (<1 second response)
- [ ] Video clip retrieval for similar situations
- [ ] Pre-match report generation (opponent tendencies)
- [ ] Post-match report generation (decision analysis)
- [ ] API for integration with other tools

---

### Phase 5: Simulation Engine (The Goal)

**This is what we're building toward.** Phases 1-4 are the foundation that makes this possible.

**Goal:** A simulation where agents can move freely, but the "physics" are learned from real football data.

**Prerequisites:** Phases 1-4 complete. Large database (5000+ possessions). Validated that our learned patterns accurately reflect reality.

---

**What We'll Have Learned (from Phases 1-4):**

| Learned Knowledge | How We Got It |
|-------------------|---------------|
| Pass success rates by situation | Similarity matching on historical passes |
| Actual pass speeds and trajectories | Measured from tracking data |
| Dribble success rates by pressure level | Outcome analysis of similar dribble situations |
| How defenders actually react | Tracked defensive movement after each action type |
| Transition probabilities | "After action X in situation Y, what happened next?" |
| Realistic state distributions | What situations actually occur vs. what's impossible |

---

**How The Simulation Works:**

Instead of: `designed rules → simulation → hope it matches reality`

We have: `real data → learned patterns → simulation that IS reality (compressed)`

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION STATE                         │
│         (22 players + ball, continuous positions)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT DECISION                            │
│     Agent considers action (pass, dribble, shot, move)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LEARNED TRANSITION MODEL                       │
│   "In similar situations, this action succeeded X% of time" │
│   "Pass speed was typically Y m/s"                          │
│   "Defenders reacted by moving Z direction"                 │
│                                                             │
│   All from real data, not designed rules                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    NEXT STATE                               │
│   Sampled from learned distributions                        │
│   Constrained to realistic states                           │
└─────────────────────────────────────────────────────────────┘
```

---

**Why Agents Can Move "Freely":**

Because we've learned the constraints from data:
- We know realistic player speeds (from tracking)
- We know realistic pass distances (from events)
- We know what defensive reactions look like (from tracking)
- We know what situations are possible vs. impossible

Agents explore within these learned bounds. They can try novel combinations, but they can't do things that never happen in real football.

---

**How Search Works (Tree Exploration):**

Not fixed depth lookahead. Variable depth tree exploration:

```
Ball reception moment
         │
         ├── Pass to Player A
         │        ├── A shoots (terminal: evaluate xG) ← STOP, evaluate
         │        ├── A passes to B
         │        │       ├── B shoots (terminal) ← STOP, evaluate
         │        │       └── B dribbles → ... ← keep exploring
         │        └── A loses ball (terminal) ← STOP, evaluate
         │
         ├── Pass to Player B (promising)
         │        └── ... ← go DEEP on this branch
         │
         ├── Dribble into pressure
         │        └── ... ← PRUNE early (low value)
         │
         └── Shot (terminal: evaluate xG) ← STOP, evaluate
```

**Search rules:**
- **Terminal states:** Shot attempt, loss of possession, goal → stop and evaluate
- **Promising branches:** Go deeper where expected value is high
- **Bad branches:** Prune early (don't waste compute on clearly bad options)
- **Uncertainty limit:** Stop if we're too far from known patterns in our database
- **Return:** The path with highest expected value of reaching scoring position

This is intelligent tree search — not uniform depth, but smart allocation of compute to promising lines.

---

**What This Enables:**

| Capability | How |
|------------|-----|
| **Tree search to scoring positions** | Explore action sequences, prune bad branches, find best path |
| **Counterfactual analysis** | "What if you'd passed left?" — explore that branch |
| **Novel tactic discovery** | Search finds high-value sequences humans haven't tried |
| **Opponent-specific planning** | Load opponent's defensive patterns, search for exploits |
| **Decision quality scoring** | Compare chosen action vs. best action from search |

---

**When To "Solve" (Frequency Progression):**

We don't simulate continuously. We solve at decision points:

| Stage | Trigger Points | Why |
|-------|----------------|-----|
| **Start** | Ball receptions only | Clear decision moment; player receives, must act |
| **Better** | Every significant ball movement | More coverage, still manageable |
| **Even better** | Whenever game state changes meaningfully | Defensive shifts, player movements |
| **Eventually** | Every 0.5 seconds (or faster) | Near-continuous analysis |

Start simple. As the system learns more and gets faster, increase frequency.

---

**Why This Is Different From Google's Approach:**

| Google Research Football | Our Approach |
|-------------------------|--------------|
| Designed physics engine | Learned physics from real data |
| Agents learn from scratch | Agents use learned human patterns as reference |
| Sim-to-real gap | No gap — simulation IS compressed reality |
| Sparse reward (goals only) | Dense signal from learned state values |
| No validation against reality | Validated against real outcomes in Phases 1-4 |

---

**Deliverables:**
- [ ] Learned transition model (pass outcomes, dribble outcomes, defensive reactions)
- [ ] Realistic state constraints (what's possible, what's not)
- [ ] Monte Carlo search using learned distributions
- [ ] Agent exploration within realistic bounds
- [ ] Sequence optimization (find best action chains)
- [ ] "What if" scenario tool for coaches

---

## What We're NOT Building Now (And Why)

| Not Building Now | Reason |
|------------------|--------|
| Discrete zone categories | Similarity algorithm handles "closeness" better than arbitrary boundaries |
| Formation classifier/labels | Behavioral features are more accurate than labels |
| Designed simulation rules | Learn transitions from data instead (Phase 5, after validation) |
| Board game abstraction | Unvalidated hypothesis; similarity approach is grounded in real data |
| Real-time in-match system | Validate offline system first; real-time adds complexity |
| Predictive model ("what will happen") | Starting with descriptive ("what has worked"); easier to validate |
| Multi-step search (initially) | Foundation must work first; simulation is Phase 5 |

---

## Open Questions (To Resolve Empirically)

1. **Feature weights:** Which features matter most for tactical similarity?
2. **Similarity threshold:** How similar is "similar enough" to draw conclusions?
3. **Sample size:** How many similar situations needed for reliable insight?
4. **Time window:** How much historical data is relevant? (Does 3-year-old data still apply?)
5. **Cross-league validity:** Do patterns transfer across leagues/competition levels?
6. **Optimal N:** How many similar situations should we retrieve? (10? 50? 100?)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Chosen features don't capture what matters | Medium | High | Start with core features, validate with coaches, iterate based on feedback |
| Not enough similar situations in database | Medium | Medium | Cast wider similarity net; aggregate more data sources |
| Coaches don't trust/use system | Medium | High | Involve coaches early; show video evidence alongside recommendations |
| Tracking data too expensive | Low | High | Prove value with available data first, then justify expanded access |
| Skill imbalance skews early results | Medium | Medium | Be explicit about limitation; address properly in Phase 3 |

---

## How Each Phase Builds To Simulation

| Phase | What We Learn | How It Powers Simulation |
|-------|---------------|-------------------------|
| **Phase 1** | Core features work; situations can be matched | Defines the state representation for simulation |
| **Phase 2** | Extended features; learned weights; what matters | Refines state representation; identifies key variables |
| **Phase 3** | Skill-adjusted patterns; separates player from tactic | Allows simulation to model different skill levels |
| **Phase 4** | Production validation; coach trust | Confirms patterns are real, not noise |
| **Phase 5** | — | **Use all learned knowledge to simulate** |

Each phase isn't just "nice to have" — it's building specific knowledge the simulation needs:

- **Pass success model:** Learned from Phase 1-2 outcome analysis
- **Dribble success model:** Learned from Phase 1-2 outcome analysis
- **Defensive reaction model:** Learned from tracking how defenders move after actions
- **State transition model:** Learned from "what state comes next" analysis
- **Realistic bounds:** Learned from what states actually occur in data

---

## Next Steps

1. **Secure tracking data access** - Required before anything else
2. **Build data ingestion pipeline** - Parse tracking + events, align by timestamp
3. **Implement core feature extraction** - Ball position, line height, compactness, pressure
4. **Create situation database** - Store feature vectors for historical possessions
5. **Build similarity search** - Nearest neighbor on normalized features
6. **Validate with coach** - Are retrieved situations tactically relevant?
7. **Track everything we learn** - Each insight becomes simulation fuel

---

*Document version: 3.0*
*Last updated: February 2026*
