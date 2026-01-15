# Demo Runbook
## Marshall University Men's Soccer — Live Demonstration

---

## Pre-Demo Checklist

### 1 Week Before
- [ ] Obtain 2-3 Marshall match clips (10-15 min each, broadcast quality)
- [ ] Run clips through pipeline to generate outputs
- [ ] Prepare sample visualizations and reports
- [ ] Test all demo equipment and connections
- [ ] Confirm attendees and room setup

### Day Before
- [ ] Load all demo materials on laptop (offline backup)
- [ ] Test screen sharing / projector connection
- [ ] Prepare printed one-pagers for attendees
- [ ] Run through demo script once

### Demo Day
- [ ] Arrive 15 min early for setup
- [ ] Test A/V before attendees arrive
- [ ] Have backup materials ready (pre-recorded video if live fails)

---

## Demo Agenda (30 Minutes)

| Time | Section | Goal |
|------|---------|------|
| 0:00 | Opening | Set context, build interest |
| 0:03 | The Problem | Why this matters now |
| 0:05 | Live Tracking Demo | Show it working on Marshall footage |
| 0:12 | Decision Engine | Show tactical analysis + predictions |
| 0:20 | Player Development | Individual metrics and trends |
| 0:25 | Q&A + Next Steps | Answer questions, propose pilot |

---

## Section Scripts

### Opening (3 min)

**Say:**
> "Thanks for making time today. I want to show you something that could change how we prepare for matches, develop players, and evaluate recruits.
>
> What I'm going to demonstrate is AI-powered player tracking — the same type of technology that Manchester City, Liverpool, and top MLS clubs pay $100,000+ per year for. We can run this on footage we already have."

**Show:** Title slide with "Professional Analytics for Marshall Soccer"

---

### The Problem (2 min)

**Say:**
> "Right now, when we review film, we're relying on what we can see and remember. We can't answer questions like:
> - How much ground did our #8 cover compared to last week?
> - Are we pressing higher than our opponents?
> - Which players are running out of gas in the 70th minute?
> - Does this recruit's movement fit our system?
>
> Commercial platforms answer these questions, but they cost $80,000 to $150,000 a year. That's not realistic for us.
>
> What I'm about to show you costs less than $500 to pilot."

**Show:** Cost comparison slide (if using slides) or just verbal

---

### Live Tracking Demo (7 min)

**This is the "wow" moment. Show real Marshall footage with tracking overlays.**

**Say:**
> "Let me show you this working on our own footage. This is from [specific match]."

**Show:**
1. **Raw footage** (10 seconds) — "Here's the original broadcast"
2. **Tracked footage** — "Now here's what the AI sees"
   - Player bounding boxes with jersey numbers
   - Movement trails
   - Ball tracking
3. **Pitch view** — "And here's the tactical view"
   - Bird's-eye 2D pitch with player dots
   - Real-time positions updating

**Highlight:**
> "Notice it's tracking all 22 players automatically. No manual tagging. It reads jersey numbers, figures out which team each player is on, and converts everything to real-world coordinates — meters on the pitch."

**Show data output:**
```
Player #7 (Marshall)
- Distance: 2,847m (first 30 min)
- Top Speed: 28.4 km/h
- Sprints: 12
- Avg Position: Right wing, attacking third
```

**Say:**
> "This data exports to spreadsheets, or we can build dashboards. Every frame, every player, every match."

---

### Decision Engine (8 min)

**This shows the "intelligence" layer — not just tracking, but understanding.**

**Say:**
> "Tracking is the foundation. But what makes this powerful is what we call the Decision Engine — AI that understands the game."

#### Part A: Tactical State Recognition

**Show:** Visualization of tactical states changing during play

**Say:**
> "The system classifies what phase of play we're in — attacking, defending, transition, set piece. Watch how it updates in real-time."

**Show classifications:**
- ATTACKING (78% confidence)
- TRANSITION_DEFENSE (91% confidence)
- DEFENDING (85% confidence)

> "Why does this matter? We can now answer: 'How much time did we spend in attacking phase vs defending?' 'How quickly do we transition?' 'Do we lose shape in transition?'"

#### Part B: Tactical Metrics

**Show:** Metrics dashboard or data table

| Metric | Marshall | Opponent |
|--------|----------|----------|
| Defensive Line Height | 42m | 38m |
| Team Compactness | 28m | 34m |
| Space Control | 54% | 46% |
| Pressing Intensity | 0.72 | 0.61 |

**Say:**
> "These metrics quantify things coaches feel but can't measure. Look — we're playing a higher line than our opponent, and we're more compact. That's intentional. But we can now track if that holds up in the 80th minute when legs get tired."

#### Part C: Prediction & Simulation

**Show:** Trajectory prediction visualization

**Say:**
> "Here's where it gets interesting for player development. The system can predict where players *should* be based on the game situation."

**Show:**
- Current player position (solid dot)
- Predicted optimal position (ghost/outline)
- Movement vector showing adjustment needed

> "If our #6 is here, but the model predicts they should be 5 meters to the left to cut the passing lane — that's a coaching point. We can show players exactly where they need to be."

**Show:** "What-if" scenario

> "We can also simulate: 'What if we pressed higher here? What if we played a back 3 instead of back 4?' The model predicts how space control and passing lanes change."

---

### Player Development Application (5 min)

**Say:**
> "Let me show you how this translates to player development."

**Show:** Individual player profile

```
PLAYER PROFILE: #10 — Alex Johnson

PHYSICAL METRICS (Season Average)
├─ Distance/Match: 9,847m (Team Avg: 9,234m) ▲
├─ Sprint Count: 34 (Team Avg: 28) ▲
├─ Top Speed: 31.2 km/h
└─ High-Intensity Distance: 2,104m

POSITIONAL METRICS
├─ Avg Position: Attacking midfield (correct)
├─ Defensive Recovery Rate: 78% (▲ from 71% last month)
├─ Space Creation: +12% vs team average
└─ Pressing Triggers: 8.4/match (target: 10)

TREND: Improving ▲
DEVELOPMENT FOCUS: Increase pressing trigger rate
```

**Say:**
> "Now we have objective data for player conversations. Instead of 'you need to press harder,' it's 'your pressing trigger rate is 8.4 per match — our target is 10. Let's look at the moments you could have pressed but didn't.'"

**Show:** Comparison view

> "For recruiting, we can evaluate prospects against our current roster. Does this kid's movement profile fit how we play? The data tells us."

---

### Q&A + Next Steps (5 min)

**Say:**
> "That's the core of what we can do. Questions?"

**Common questions to prepare for:**

| Question | Answer |
|----------|--------|
| "How accurate is it?" | "We'll validate against manual spot-checks. Pro systems claim 95%+ accuracy; ours should be comparable." |
| "What footage do we need?" | "Broadcast quality, wide-angle preferred. What we already have from ESPN+/conference feeds works." |
| "How long to process a match?" | "About 60-90 minutes of compute time per match. Results ready same day." |
| "Who operates it?" | "After training, 15-30 min of staff time per match to review and export data." |
| "What's the cost?" | "Pilot is ~$500 for 5 matches. Full season would be $8-10K — vs $100K+ for commercial." |

**Close with:**
> "I'd like to propose a pilot: 5 matches over the next few weeks. We'll deliver full tracking data, tactical analysis, and player reports. If it's useful, we discuss a full season deployment. If not, we're out $500 and some time.
>
> Can we schedule those 5 matches?"

---

## Demo Materials Checklist

### Video Assets
- [ ] Marshall match clip #1 (with tracking overlay)
- [ ] Marshall match clip #2 (raw + tracked comparison)
- [ ] Tactical view animation (2D pitch)
- [ ] Trajectory prediction visualization

### Data Assets
- [ ] Sample player profile PDF
- [ ] Tactical metrics comparison table
- [ ] Full match data export (JSON/CSV)

### Backup Materials
- [ ] Pre-recorded demo video (in case live fails)
- [ ] Static screenshots of key visualizations
- [ ] Printed one-page executive summary

---

## Post-Demo Actions

Immediately after:
- [ ] Send thank-you email with executive summary attached
- [ ] Note any specific requests or concerns raised
- [ ] Confirm pilot match selection

Within 48 hours:
- [ ] Deliver any additional materials requested
- [ ] Send pilot program proposal with timeline
- [ ] Schedule pilot kickoff meeting

---

## Demo Environment Setup

### Required Software
```bash
# Ensure pipeline is ready
cd /home/user/football
python -m src.main --help  # Verify installation

# Pre-process demo clips
python -m src.main data/demo/marshall_clip1.mp4 \
    --config config/config.yaml \
    --visualize \
    --output data/demo/output/
```

### Hardware Requirements
- Laptop with demo materials loaded
- HDMI/USB-C adapter for projector
- Backup on USB drive
- Mobile hotspot (backup internet)

### Room Setup
- Screen/projector visible to all attendees
- Seating for 4-8 people
- Whiteboard for follow-up discussion
