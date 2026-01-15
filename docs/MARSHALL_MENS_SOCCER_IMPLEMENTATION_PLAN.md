# Marshall Men's Soccer: Player Tracking & Decision Engine

## Part 1: What The Tracking System Gives You

Run a match video through the system:
```bash
python -m src.main match_video.mp4 --visualize
```

### Raw Output: Player Positions Every Frame

```json
{
  "frame": 1500,
  "timestamp": 60.0,
  "players": [
    {"track_id": 3, "team": "home", "x": 25.4, "y": 18.2, "speed_kmh": 12.3},
    {"track_id": 7, "team": "home", "x": 45.2, "y": 23.1, "speed_kmh": 28.5},
    {"track_id": 11, "team": "away", "x": 38.7, "y": 31.4, "speed_kmh": 6.1},
    {"track_id": 14, "team": "away", "x": 42.1, "y": 12.8, "speed_kmh": 22.7}
  ],
  "ball": {"x": 44.8, "y": 24.2}
}
```

### Per-Player Metrics (End of Match)

| Player | Distance | Max Speed | Sprints | High-Intensity Distance |
|--------|----------|-----------|---------|------------------------|
| Track 3 | 10.2 km | 28.4 km/h | 18 | 1.8 km |
| Track 7 | 11.8 km | 31.5 km/h | 26 | 2.4 km |
| Track 11 | 9.1 km | 25.2 km/h | 12 | 1.2 km |

### Visualizations

- Pitch animation showing all player movements
- Heatmaps of player positions
- Team shape over time

---

## Part 2: How The Decision Engine Uses This Data

The tracking data alone tells you WHERE players are. The decision engine tells you WHAT IT MEANS.

### Concept: Elimination

A defender is **eliminated** when:
1. The ball is past them (closer to their goal than they are)
2. They cannot recover in time to make an effective intervention

**Raw tracking data:**
```
Ball: x=45, y=24
Defender #14: x=42, y=12 (speed: 22.7 km/h)
Attacker #7: x=45, y=23 (speed: 28.5 km/h, has ball)
```

**Decision engine analysis:**
```
Defender #14 is ELIMINATED
- Ball is 3m ahead of defender (x: 45 vs 42)
- Defender's max recovery speed: 26 km/h
- Time to intercept path: 2.1 seconds
- Attacker reaches danger zone in: 1.4 seconds
- Defender CANNOT recover → eliminated
```

### What This Means for Coaches

**Without decision engine:**
"Here's where everyone was at minute 60."

**With decision engine:**
"At minute 60, their right back was eliminated. This is the 4th time this half. He's getting eliminated because he's stepping up to press but can't recover - his max speed is 26 km/h, your winger is hitting 31 km/h."

---

## Part 3: The Progression

### Stage 1: Tracking Data (Available Now)

**What you see:**
- Player positions every frame
- Individual speeds and distances
- Team formations

**Coach use:**
- Review positioning
- Track physical output
- See movement patterns

### Stage 2: Decision Engine Analysis (After Integration)

**What you see:**
- Which defenders are eliminated each frame
- Danger level of each attacking situation
- Defensive structure quality score

**Coach use:**
- Identify defensive breakdowns
- See which players are most vulnerable
- Quantify how dangerous chances actually were

### Stage 3: Opponent Scouting (With Data Accumulation)

**What you see:**
- Opponent player speed profiles
- Their typical elimination patterns
- Structural weaknesses by position

**Coach use:**
- Target slow defenders with fast attackers
- Know when their structure typically breaks
- Prepare specific matchup advantages

---

## Example: Full Analysis Flow

### 1. Raw Tracking Data
```
Frame 4521 (75:21)
Ball position: (48.2, 22.1)
Your LW (#7): (47.8, 21.5) - 29.2 km/h - has ball
Their RB (#2): (44.1, 18.3) - 8.4 km/h - jogging
Their RCB (#4): (38.2, 26.7) - 4.2 km/h - standing
```

### 2. Decision Engine Output
```
ELIMINATION STATUS:
- Their RB (#2): ELIMINATED
  → Ball is 4.1m past him
  → Recovery time needed: 2.8s
  → Your attacker reaches box in: 1.9s
  → Cannot recover

- Their RCB (#4): NOT ELIMINATED (yet)
  → Can cover if moves now
  → Window: 1.2 seconds

GAME STATE SCORE: 0.78 (High danger)
RECOMMENDATION: Attack now - 1v1 vs goalkeeper if RCB doesn't shift
```

### 3. Accumulated Pattern (After Multiple Matches)
```
OPPONENT RB (#2) PROFILE:
- Max speed: 26.1 km/h (below average)
- Gets eliminated: 5.2 times per match
- Most vulnerable: When pressing high
- Recovery runs: Slow (18.4 km/h average)

YOUR LW (#7) vs THEIR RB (#2):
- Speed advantage: +5.4 km/h
- Historical eliminations caused: 8 in 2 matches
- Success rate when isolated 1v1: 73%

TACTICAL RECOMMENDATION:
Play direct balls behind RB when he steps to press.
Your winger wins that race every time.
```

---

## Development Path

| Phase | What's Added | Value to Coaches |
|-------|--------------|------------------|
| **Now** | Position data + physical metrics | See where everyone is, who worked hardest |
| **+2 weeks** | Elimination detection | See when/why defensive breakdowns happen |
| **+1 month** | Per-player opponent profiles | Know opponent speeds, find mismatches |
| **+1 season** | Pattern database | "Last time we did X, they did Y" |

---

## What Needs to Happen

### Already Done (Run Today):
- [x] Track all players frame-by-frame
- [x] Calculate speeds, distances, sprints
- [x] Export to JSON/CSV
- [x] Generate visualizations

### Integration Work (2-3 Days Dev):
- [ ] Connect decision engine to main pipeline
- [ ] Add elimination data to output JSON
- [ ] Add game state scores to output

### Future Development:
- [ ] Report generator (auto-create scouting documents)
- [ ] Matchup comparison tool
- [ ] Historical database queries

---

## Summary

**Tracking System = WHERE players are**
- Positions, speeds, distances
- Raw data for any analysis

**Decision Engine = WHAT IT MEANS**
- Who's eliminated, who's vulnerable
- How dangerous is this situation
- Which matchups favor you

The tracking data feeds the decision engine. More matches filmed = more opponent data = smarter tactical recommendations.
