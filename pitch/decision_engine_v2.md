# Decision Engine
## Ball Progression & Defensive Weakness Analysis

---

## Core Concept

The Decision Engine answers one question: **"Where should the ball go next, and what are the odds of getting it there?"**

```
INPUT                           OUTPUT
─────                           ──────
22 player positions      →      Defensive weaknesses (gaps)
Ball position            →      Optimal progression paths
Historical success rates →      Success probability per action
                         →      xG value of target zones
                         →      Risk/reward score
```

---

## Visualization: Keep It Simple

Just dots on a pitch. Nothing fancy.

```
┌─────────────────────────────────────────┐
│                                         │
│         ○           ●                   │
│    ○         ●            ●      ○      │
│                                         │
│       ○        ◉        ●         ○     │
│               ball                      │
│   ○       ●        ●           ○        │
│                                         │
│        ○      ●          ○              │
│             ○     ●   ○                 │
│                  ○                      │
│                                         │
└─────────────────────────────────────────┘

● = Team A (home)
○ = Team B (away)
◉ = Ball

Frame updates: positions move based on tracking coordinates
That's it.
```

---

## What the Engine Actually Calculates

### 1. Defensive Weaknesses

**Find the gaps in the opponent's shape.**

For every frame, calculate:
- Distance between each pair of defenders
- Coverage zones (where can each defender reach in 1s, 2s, 3s?)
- Gaps = areas outside all defender coverage zones

```
COVERAGE MODEL

Each defender has an interception range:

    Defender at (x, y)
    Speed: 6 m/s
    Reaction time: 0.3s

    Can reach:
    - In 1s: ~4m radius (after reaction)
    - In 2s: ~10m radius
    - In 3s: ~16m radius

    ┌─────────────────┐
    │    ╭─────╮      │
    │   ╱       ╲     │  ← 2s range
    │  │    ○    │    │  ← defender
    │   ╲       ╱     │
    │    ╰─────╯      │
    └─────────────────┘
```

**Gap Detection:**
```
When coverage zones don't overlap = GAP

    ╭───╮       ╭───╮
   ╱     ╲     ╱     ╲
  │   ○   │ █ │   ○   │   █ = exploitable gap
   ╲     ╱     ╲     ╱
    ╰───╯       ╰───╯

Gap location: between CB and RB
Gap size: 8m
Time to close: 1.4s
```

### 2. xG Zones

**Map the pitch by expected goal value.**

Based on historical shot data, every location has an xG value if a shot is taken from there:

```
xG ZONE MAP

┌─────────────────────────────────────────┐
│                 GOAL                    │
│              ┌──────┐                   │
│           0.35  0.42  0.35              │  ← High xG (inside box)
│        0.18    0.25    0.18             │
│      0.08      0.12      0.08           │  ← Medium xG
│    0.03        0.05        0.03         │
│  0.01          0.02          0.01       │  ← Low xG
│                                         │
└─────────────────────────────────────────┘

Numbers = probability of goal if shot taken from that zone
```

**The goal:** Progress the ball into higher xG zones.

### 3. Ball Progression Paths

**Calculate how to move the ball from current position to high xG zones.**

For each potential next action (pass, dribble, through ball):

```
PROGRESSION OPTIONS FROM CURRENT POSITION

Current ball position: (35, 40) — midfield, left side

Option A: Short pass to #8
├── Target: (40, 38)
├── Distance: 5.4m
├── Defender pressure: LOW (nearest defender 8m away)
├── Pass success probability: 89%
├── xG zone change: 0.01 → 0.02 (+0.01)
└── Risk/Reward: LOW RISK, LOW REWARD

Option B: Through ball to #9
├── Target: (62, 35) — edge of box
├── Distance: 27m
├── Defender pressure: HIGH (passes through coverage zone)
├── Interception probability: 34%
├── Pass success probability: 66%
├── xG zone change: 0.01 → 0.15 (+0.14)
└── Risk/Reward: HIGH RISK, HIGH REWARD

Option C: Switch to #7 (right wing)
├── Target: (38, 58)
├── Distance: 18m
├── Defender pressure: MEDIUM
├── Pass success probability: 78%
├── xG zone change: 0.01 → 0.03 (+0.02)
└── Risk/Reward: MEDIUM RISK, LOW REWARD
```

### 4. Success Probability Model

**Based on real data, not guesses.**

Factors that determine if a pass/action succeeds:

| Factor | Impact |
|--------|--------|
| Pass distance | Longer = lower success |
| Defender proximity to target | Closer = lower success |
| Defender proximity to ball | Pressure reduces accuracy |
| Receiver movement | Moving away = harder |
| Pass angle | Backwards safer than forward |
| Player's historical success rate | Individual skill matters |

**Model:**
```
P(success) = f(distance, pressure, angle, player_skill, receiver_position)

Trained on historical match data:
- Pass completion rates by distance
- Duel success rates by player
- Interception rates by defender position
```

### 5. Interception Range

**Where can defenders intercept?**

```
INTERCEPTION MODEL

Ball traveling from A to B:

    A ───────────────→ B
           │
           ○ defender

Time for ball to reach point X: t_ball = distance / ball_speed
Time for defender to reach X: t_defender = distance / run_speed + reaction

If t_defender < t_ball at any point along path → INTERCEPTION RISK
```

**Calculate for every potential pass:**
- Draw line from ball to target
- Check if any defender can reach any point on that line before the ball
- Output: interception probability

### 6. Risk/Reward Score

**Expected value of each action.**

```
Expected Value = P(success) × Value_gained - P(failure) × Cost

Where:
- P(success) = pass completion probability
- Value_gained = xG increase from reaching target zone
- P(failure) = 1 - P(success)
- Cost = xG risk if turnover (opponent counter potential)
```

**Example calculation:**

```
Through ball to striker in box:

P(success) = 0.55 (55% chance of completion)
Value_gained = 0.18 xG (shooting opportunity)
P(failure) = 0.45
Cost = 0.08 xG (dangerous counter if intercepted)

EV = 0.55 × 0.18 - 0.45 × 0.08
EV = 0.099 - 0.036
EV = +0.063 xG

Decision: POSITIVE expected value → worth attempting
```

---

## Engine Outputs

### Per Frame Analysis

```json
{
  "frame": 12345,
  "timestamp": 34.21,
  "ball_position": {"x": 35, "y": 40},
  "ball_carrier": {"player_id": 8, "team": "home"},

  "defensive_gaps": [
    {
      "location": {"x": 55, "y": 32},
      "size_meters": 8.2,
      "time_to_close": 1.4,
      "exploitable": true
    }
  ],

  "progression_options": [
    {
      "action": "pass",
      "target_player": 9,
      "target_position": {"x": 62, "y": 35},
      "success_probability": 0.66,
      "interception_risk": 0.34,
      "xg_current": 0.01,
      "xg_target": 0.15,
      "xg_gain": 0.14,
      "turnover_cost": 0.08,
      "expected_value": 0.063,
      "recommendation": "HIGH_VALUE"
    },
    {
      "action": "pass",
      "target_player": 6,
      "success_probability": 0.91,
      "xg_gain": 0.01,
      "expected_value": 0.008,
      "recommendation": "SAFE"
    }
  ],

  "best_option": {
    "action": "pass",
    "target": 9,
    "reason": "Highest expected value, gap exploitable"
  }
}
```

### Match Summary

After processing a full match:

```
BALL PROGRESSION ANALYSIS

Total possessions: 87
Progressed to final third: 34 (39%)
Progressed to box: 12 (14%)
Shots generated: 8

Opportunities missed:
- 14 instances where high-value through ball was available but not played
- Average EV of missed opportunities: +0.07 xG each
- Total missed xG: ~1.0 xG

Key findings:
- Gap between CB-RB exploited only 3 of 11 times it was available
- Team chose safe pass (EV < 0.02) when better option existed 67% of time
- Highest value chances came from quick transitions (< 4 seconds)
```

---

## Coaching Application

### Pre-Match: Opponent Weakness Map

```
OPPONENT DEFENSIVE WEAKNESS REPORT

Based on 3 matches analyzed:

Primary weakness: Channel between #4 (CB) and #2 (RB)
├── Gap appears: 12.3 times/match average
├── Size when open: 7-10m
├── Best trigger: When LW receives ball, RB steps to press
├── Exploited against them: 4 times → 2 shots, 0.31 xG

Secondary weakness: High line vulnerable to balls over top
├── Line height: 44m average (high)
├── Recovery speed: Slow (#4 runs 5.8 m/s)
├── Success rate of through balls vs them: 61%
```

### In-Match: Decision Support

```
LIVE ANALYSIS (if processing in real-time)

Current situation:
- Ball at (38, 42) with #8
- Gap detected between CB-RB (7m, closing in 1.2s)
- Through ball to #9 has +0.09 EV
- Window: NOW

[System could flag this to analyst on sideline]
```

### Post-Match: Decision Quality Review

```
PLAYER DECISION ANALYSIS: #8 (Central Midfielder)

Possessions: 67
Decisions analyzed: 54 (excluding simple layoffs)

Made optimal choice: 31 (57%)
Made safe choice when better existed: 18 (33%)
Made risky choice when safe was better: 5 (10%)

xG left on table: 0.42 (from suboptimal decisions)

Development focus:
- Recognize through ball opportunities faster
- Trust #9's runs more (success rate when played: 68%)
```

---

## What This Is NOT

- ❌ Fancy 3D visualizations
- ❌ Complex UI with 10 different layers
- ❌ Real-time player tracking hardware
- ❌ Magic predictions with no basis

## What This IS

- ✅ Dots on a pitch (simple visualization)
- ✅ Math-based gap detection (defender coverage models)
- ✅ Probability models trained on real data
- ✅ Expected value calculations for each decision
- ✅ Actionable insights for coaching

---

## Data Requirements

To make this work:

| Data | Source | Purpose |
|------|--------|---------|
| Player positions | Tracking system | Gap detection, coverage zones |
| Ball position | Tracking system | Current state |
| Historical pass data | Wyscout/StatsBomb | Success probability model |
| xG model | Public models or trained | Zone values |
| Player speed data | Tracking system | Interception range |

---

## Summary

The Decision Engine is a calculator, not a crystal ball.

**Input:** Where everyone is right now
**Math:** Coverage zones, gaps, distances, probabilities
**Output:** "Here's where the defense is weak, here's the best path to goal, here's your odds"

Coaches still make decisions. The engine just shows them what the numbers say.
