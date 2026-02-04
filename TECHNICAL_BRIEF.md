# Tracking + Decision Engine — Technical Brief

This document explains what's built, what works, and what needs to happen next.

---

## Overview

Two layers that work together:

1. **Tracking Layer** — Extract player/ball coordinates from video
2. **Decision Engine Layer** — Analyze those coordinates tactically

```
Video → Detection → Tracking → Calibration → Coordinates → Decision Engine → Analysis
```

---

## Current State: What's Built

### Codebase Size
- **~36,000 lines** of Python across detection, tracking, calibration, decision engine, training infrastructure
- Modular architecture using PyTorch

### What's Working Now

| Component | Status | What It Does |
|-----------|--------|--------------|
| Player Detection (YOLO) | ✅ Working | Finds players in frames, confidence filtering |
| Ball Detection | ✅ Working | Finds ball with temporal consistency, interpolates when lost |
| Team Classification | ✅ Working | Clusters jersey colors (home/away/referee) |
| ByteTrack | ✅ Working | Maintains player IDs frame-to-frame |
| Manual Calibration | ✅ Working | Click pitch points to map pixels → coordinates |
| Decision Engine UI | ✅ Working | Interactive tactical board with real-time analysis |

### What's Implemented But Needs Training/Testing

| Component | Status | What's Missing |
|-----------|--------|----------------|
| Auto Calibration | Implemented | Needs testing on Wyscout footage |
| DETR (Transformer Detection) | Implemented | Needs pre-trained weights |
| Ball 3D Tracking | Implemented | LSTM model needs training |
| Player Re-ID | Implemented | OSNet weights needed |
| Extrapolation (Baller2Vec) | Implemented | Model weights needed |
| GNN Tactical Analysis | Implemented | Needs labeled tactical data |

### What's Not There Yet

| Component | Status | Dependency |
|-----------|--------|------------|
| Camera angle handling | Gap | Auto-calibration must work reliably |
| 90-minute tracking | Gap | Re-ID and extrapolation must work |
| Game model labeling | Gap | Manual process to define |
| Outcome-based learning | Gap | Needs tracking data connected to results |

---

## Layer 1: Tracking — Detailed

### Detection

**Primary: YOLOv8**
- Detects players and ball in each frame
- Works well on clear, unoccluded players
- Struggles with crowded areas, overlapping players

**Backup: DETR (Transformer)**
- Better for crowded scenes (uses attention, not anchor boxes)
- Architecture implemented, needs training

**Ball Detection**
- Specialized detector with temporal consistency
- Interpolates when ball is lost (up to 5 frames)
- Validates velocity to prevent jitter

### Tracking (ByteTrack)

- Maintains consistent player IDs across frames
- Works on stable footage
- Loses identity through occlusions and camera cuts
- Not reliable across full 90 minutes yet

### Calibration (Pixel → Pitch Coordinates)

**The Main Challenge**

Wyscout footage uses wide-angle cameras that shift during play. We need to know where each pixel corresponds to on the pitch.

**Manual Calibration (Working)**
- Click 4+ points on the pitch (corners, penalty spots, etc.)
- System computes transformation matrix
- Works when camera angle is stable

**Auto Calibration (Implemented)**
- Neural network detects pitch keypoints automatically
- RANSAC-based homography estimation
- Quality metrics (reprojection error, inlier ratio)
- Needs testing on real Wyscout footage to validate

**Field Model**
- 57+ keypoints defined (corners, boxes, circles, spots)
- Standard FIFA dimensions (105m x 68m)

### Supporting Components

**Rotation Handler**
- Tracks camera state changes
- Adaptive homography management
- Handles rotating camera footage

**Player Re-Identification**
- OSNet embeddings for appearance matching
- Jersey number detection and recognition
- Cross-match identification
- Needed for: recovering identity after occlusions, camera cuts

**Extrapolation (Baller2Vec)**
- Predicts player positions when off-screen
- Transformer architecture with team coordination
- Kalman filter fallback
- Needed for: full-field tracking from broadcast footage

---

## Layer 2: Decision Engine — Detailed

### Interactive Tactical Board (Working)

A web-based UI where you can:
- Drag players to any position
- Move the ball
- Get real-time tactical analysis

**What it calculates:**

1. **xG from any position**
   - Based on distance to goal, shooting angle
   - Accounts for shot blocking by defenders

2. **Pass options**
   - Evaluates every teammate as a target
   - Calculates interception probability using physics
   - Ball travel time vs defender arrival time

3. **Through ball detection**
   - Where can attackers run?
   - Can the ball reach the space before defenders?
   - Accounts for offside

4. **Dribble options**
   - Can the player beat the nearest defender?
   - Time to dribble vs time for defender to close

5. **Gaps in defensive lines**
   - Identifies spaces between defenders
   - Calculates time to exploit vs time to close

6. **Option ranking**
   - HIGH_VALUE, SAFE, MODERATE, LOW_VALUE, AVOID
   - Shows best action and why

7. **Action simulation**
   - Click "play" on an option
   - See positions update
   - Get new analysis

### Physics Model

Calibrated to real player data:

| Parameter | Value | Source |
|-----------|-------|--------|
| Sprint: 0-10m | 1.8 seconds | Elite player data |
| Top speed | 8.33 m/s (30 km/h) | Elite player data |
| Reaction time | 0.3 seconds | Research literature |
| Ball deceleration | 0.5 m/s² | Grass friction |
| Pass speed | 5-20 m/s | Soft to hard pass |

**Interception calculation:**
```
1. Find closest point on ball path to defender
2. Calculate defender run distance (minus 1m reach)
3. Add reaction time (0.3s) + acceleration curve time
4. Compare to ball travel time (with deceleration)
5. If defender arrives first → interception
```

### Core Modules

**Elimination Calculator**
- Determines which defenders can affect the play in time
- Not just "goal-side" — who can actually intervene
- Accounts for momentum, reaction time, sprint speed

**State Scorer**
- Composite score for any game moment
- Components: elimination ratio, proximity, angle, density, compactness
- Used to compare "what if we played here instead?"

**Defensive Force Model**
- Models defensive positioning as attraction/repulsion forces
- Ball attraction (pressing), goal protection, zone coverage, marking, spacing
- Tunable weights to model different defensive styles

**Block Models**
- LOW, MID, HIGH block configurations
- Line heights, compactness targets, press triggers
- Can model opponent defensive setups

---

## The Gap: Physics vs Reality

### What the engine does now
Evaluates based on physics and geometry. "Can the ball reach the target before defenders intercept?"

### What it can't do yet
Know what actually works in real football. The math might say a through ball is available, but:
- Maybe that pass gets read and cut out 80% of the time
- Maybe the receiver always takes a bad touch under pressure
- Maybe the defender anticipates and intercepts even though physics says he can't

### How we close the gap
**Train on real games.** Feed the engine tracking data from Marshall matches. Label moments according to our game model. Connect to outcomes — did the pass work? did we create a chance? did we lose the ball?

Over time, the engine learns:
- What options actually succeed vs just look good on paper
- How Marshall players execute in different situations
- Patterns that lead to good outcomes for our style

---

## Development Plan

### Phase 1: Tracking Validation (Month 1-2)

**Goal:** Confirm we can get accurate coordinates from Wyscout footage.

**Tasks:**
1. Test auto-calibration on Wyscout video
   - Does it find pitch keypoints reliably?
   - How does it handle camera angle changes?
   - What's the reprojection error?

2. Run full tracking pipeline on a test match
   - Detection → Tracking → Calibration → Coordinates
   - Visual validation: do the pitch positions look right?

3. Compare against GPS
   - For matches where we have both video and GPS
   - Calculate position error
   - Identify where tracking fails

**Success criteria:**
- Average position error < 2m
- Track maintained for 80%+ of each player's on-ball time
- Calibration handles camera shifts without manual intervention

### Phase 2: Tracking Reliability (Month 2-3)

**Goal:** Handle the hard cases — occlusions, camera cuts, crowded areas.

**Tasks:**
1. Train/integrate player re-identification
   - OSNet for appearance embeddings
   - Jersey number recognition
   - Test identity recovery after occlusions

2. Improve ball tracking
   - Train Ball 3D LSTM if needed
   - Validate ball position accuracy

3. Test on multiple matches
   - Different opponents, different camera angles
   - Build confidence in reliability

**Success criteria:**
- Identity maintained through camera cuts
- Ball tracked accurately during passes and shots
- System works on any Wyscout match without manual tweaking

### Phase 3: Game Model Labeling (Month 3-4)

**Goal:** Create the vocabulary for training the decision engine.

**Tasks:**
1. Define moment types based on Marshall game model
   - BGZ buildup, FGZ attack, High Loss counter-press, etc.
   - Pressing triggers, structure checks

2. Build labeling workflow
   - Tool to mark moment type at each frame/sequence
   - Connect to tracking data

3. Label initial dataset
   - 5-10 matches with moment labels
   - Focus on key sequences (buildups, transitions, chances)

**Output:**
- Labeled tracking data: position coordinates + moment type + outcome

### Phase 4: Decision Engine Training (Month 4-5)

**Goal:** Engine learns from real outcomes, not just physics.

**Tasks:**
1. Connect tracking data to decision engine
   - Feed real coordinates (not manual drag/drop)
   - Analyze what options were available at each moment

2. Compare predictions to outcomes
   - Engine says "pass to #7 is best option"
   - What did we actually do? What happened?

3. Adjust model based on reality
   - Which predictions were right/wrong?
   - Learn patterns: "against low blocks, switches work better than physics predicts"

**Success criteria:**
- Engine recommendations align with successful outcomes
- Face validity: coaches agree with engine analysis
- Measurable improvement in prediction accuracy over time

### Phase 5: Integration with Hub (Month 5+)

**Goal:** Tracking and decision engine data flows into the Hub.

**Tasks:**
1. Export tracking data to Hub-compatible format
2. Enable queries across tracking + Wyscout + GPS
3. Build visualizations for coaches

---

## Technical Dependencies

### Required Pre-trained Weights

| Model | Purpose | Source |
|-------|---------|--------|
| YOLOv8x | Player/ball detection | Ultralytics (available) |
| OSNet | Player re-identification | torchreid (available) |
| Pitch keypoint detector | Auto-calibration | Train on SoccerNet |
| Ball 3D LSTM | 3D ball tracking | Train on synthetic + real |
| Baller2Vec | Trajectory prediction | Train on tracking data |

### Hardware

- GPU for inference (RTX 3080+ recommended)
- Storage for video files (~10GB per match)
- Processing: ~10-15 FPS real-time, faster in batch

### Data Required

| Data Type | Purpose | Source |
|-----------|---------|--------|
| Wyscout video | Input for tracking | Existing subscription |
| GPS data | Validation ground truth | Existing from Catapult |
| Game model moments | Training labels | Manual labeling |
| Match outcomes | Learning signal | Wyscout events + video |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Auto-calibration fails on Wyscout | Can't scale to many matches | Fall back to semi-manual; invest in better keypoint model |
| Tracking loses identity too often | Data not usable | Focus on key sequences only; use re-ID aggressively |
| Labeling takes too long | Can't train engine | Start simple (fewer moment types); build efficient tools |
| Physics predictions don't match reality | Engine gives bad advice | Expected — that's why we train on real data |

---

## Summary

**What works now:**
- Detection and basic tracking
- Manual calibration
- Interactive decision engine with physics-based analysis

**Critical path:**
1. Validate tracking accuracy against GPS
2. Get auto-calibration working reliably
3. Label matches with game model moments
4. Train engine on real outcomes

**The end goal:**
An engine that can analyze any moment in a Marshall game, understand what options were available, evaluate them based on what actually works for us (not just physics), and show — player by player — what's working and what needs adjustment.
