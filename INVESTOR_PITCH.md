# Tactical Intelligence Platform
## Investor Pitch

---

## The Problem

Soccer analytics is stuck at the event level.

Current tools tell you *what happened* — passes completed, shots taken, possession percentage. They don't tell you *why* it happened or *what should have happened*.

**The gap:**
- Wyscout, Hudl, InStat: Event data and video tagging
- StatsBomb, Opta: Advanced metrics on events (xG, xA)
- Second Spectrum, SkillCorner: Tracking data (positions over time)

None of them answer the coaching question: **"Was that the right decision?"**

A midfielder completes a pass. Event data says: pass successful. But was it the *best* pass? Was there a through ball that would have created a goal? Was the forward making a run the midfielder didn't see?

**Coaches watch film for hours trying to answer these questions manually.** They can't quantify it. They can't compare it across games. They can't give players objective feedback on decision quality.

---

## The Solution

A tactical intelligence platform that evaluates every decision, not just every event.

**What we do differently:**

| Current Tools | Our Platform |
|---------------|--------------|
| Records what happened | Evaluates what should have happened |
| Event-level (pass, shot, tackle) | Decision-level (was that the best option?) |
| Aggregate stats (completion %) | Moment-specific analysis |
| Generic metrics | Adapts to team's tactical philosophy |
| Requires analyst interpretation | Produces actionable coaching insights |

**Core capability:** For any frame of video, we identify all available options (pass targets, dribble lanes, shooting angles), calculate the expected value of each, and compare against what actually happened.

---

## How It Works

### Layer 1: Tracking
Computer vision extracts player and ball positions from video. Any video — broadcast, tactical cam, phone footage.

### Layer 2: Decision Engine
Physics-based analysis of every frame:
- Which defenders are eliminated (can't reach the ball in time)?
- What passing lanes are open?
- What's the expected value of each option?
- What option should have been chosen?

### Layer 3: Game Model Integration
The engine adapts to any tactical philosophy:
- High press team? Weights counter-pressing opportunities higher
- Possession-based? Values ball retention in buildup
- Direct play? Prioritizes vertical progression

**The same engine, different configurations.** A team defines their philosophy, and the engine evaluates execution against it.

### Layer 4: Outcome Learning
Over time, the engine learns what actually works for each team:
- Physics says Option A is best
- Reality shows Option A fails 60% of the time for this team
- Engine adjusts: recommend Option B in similar situations

**Personalized intelligence** that improves with every game analyzed.

---

## Market Opportunity

### Primary Market: Professional & Semi-Pro Clubs

**Total addressable market:**
- ~500 professional clubs in top European leagues
- ~150 MLS, Liga MX, other Americas leagues
- ~200 top clubs in other regions
- ~3,000 second/third tier professional clubs

**Serviceable market (Year 1-3):**
- English-speaking markets: MLS, Championship, League One/Two, USL
- ~400 clubs, $50-200K/year potential per club

### Secondary Market: College Programs

**Total addressable market:**
- 206 NCAA Division I men's programs
- 333 NCAA Division I women's programs
- 200+ Division II programs with competitive budgets

**Serviceable market:**
- Power 5 + top 50 programs
- ~100 programs, $20-50K/year potential

### Tertiary Market: Academies & Youth Development

**Long-term opportunity:**
- Professional club academies (200+)
- Elite youth clubs (1,000+)
- Scaled-down product at $5-15K/year

---

## Competitive Landscape

| Company | What They Do | Limitation |
|---------|--------------|------------|
| **Hudl** | Video platform, basic tagging | No decision analysis |
| **Wyscout** | Scouting database, event data | Events only, no spatial analysis |
| **StatsBomb** | Advanced event metrics | Still event-level, no real-time tracking |
| **Second Spectrum** | Tracking data | Raw data, requires interpretation |
| **SkillCorner** | Broadcast tracking | Data provider, not insight provider |

**Our differentiation:**
1. **Decision-level analysis** — not just "pass completed" but "was it the right pass"
2. **Game model integration** — adapts to any tactical philosophy
3. **Outcome learning** — improves with each team's data
4. **Actionable output** — player-specific coaching insights, not raw data

**Moat:** The outcome learning creates a flywheel. More games analyzed → better recommendations → more value → more teams adopt → more data → better recommendations.

---

## Business Model

### Subscription Tiers

| Tier | Price/Year | Includes |
|------|-----------|----------|
| **Starter** | $25,000 | Post-match analysis, 30 games/season, team-level insights |
| **Professional** | $75,000 | All matches, player-level reports, opponent modeling |
| **Elite** | $150,000 | Real-time analysis, custom integrations, dedicated support |

### Revenue Projections

| Year | Teams | ARR | Notes |
|------|-------|-----|-------|
| Year 1 | 5 | $200K | Pilot customers, prove value |
| Year 2 | 25 | $1.2M | Expand in college + MLS |
| Year 3 | 75 | $4.5M | Add European leagues |
| Year 5 | 200 | $15M | Market leader in decision analytics |

### Unit Economics

- **CAC:** $15-25K (pilot program + sales cycle)
- **LTV:** $150K+ (3-5 year retention expected)
- **Gross margin:** 80%+ (software, minimal marginal cost)
- **Payback:** < 12 months

---

## Traction

### Current State

**Built:**
- 36,000+ lines of production code
- Working decision engine with five core modules
- Player/ball detection and tracking pipeline
- Interactive tactical analysis UI

**Pilot customer: Marshall University Men's Soccer**
- Division I program, Sun Belt Conference
- Using platform to analyze game model execution
- Providing validation feedback for product development

### Roadmap

| Milestone | Status |
|-----------|--------|
| Decision engine core modules | Complete |
| Tracking pipeline | In development |
| Marshall pilot | Active |
| Face validity with coaching staff | Next phase |
| Outcome learning system | Designed, not built |
| First paying customer | Target: post-validation |

---

## Team

*[To be filled in with actual team details]*

**What we need:**
- Technical: ML/computer vision, backend engineering
- Domain: Soccer coaching/analysis background
- Business: Sales to sports organizations

---

## The Ask

**Raising:** $500K seed round

**Use of funds:**
- 60% Engineering (tracking reliability, outcome learning)
- 20% Pilot expansion (3-5 additional teams)
- 15% Sales/marketing (conference presence, demos)
- 5% Operations

**Milestones for seed:**
- Tracking accuracy validated
- 3+ pilot teams with positive feedback
- First revenue (even small)
- Clear path to Series A metrics

---

## Why Now

1. **Tracking technology matured** — YOLO, ByteTrack, and transformer models make broadcast-quality tracking feasible
2. **Analytics adoption accelerating** — Even mid-tier clubs now have data staff
3. **Decision analysis is the next frontier** — Event data is commoditized, everyone has it
4. **AI/ML infrastructure cheap** — GPU compute, model training, deployment all accessible
5. **Coaching demands growing** — Player salaries up, margins thin, every edge matters

---

## Vision

**Year 1:** Prove decision-level analysis delivers coaching value

**Year 3:** Standard tool for serious programs — "We use tactical intelligence the way we use GPS tracking"

**Year 5:** The way soccer understands decisions changes. Not "did the pass work" but "was it the right pass given all options"

**Long-term:** Every player gets objective feedback on every decision. Development becomes quantified. Scouting evaluates decision quality, not just physical output.

---

## Summary

- **Problem:** Analytics stuck at event level, coaches can't evaluate decisions
- **Solution:** Platform that analyzes every option, adapts to any game model
- **Market:** $500M+ TAM across professional and college soccer
- **Traction:** 36K lines of code, working engine, active pilot
- **Ask:** $500K to validate tracking and expand pilots
- **Vision:** Decision analytics becomes standard — we define the category

---

*Contact: [To be added]*

*More details: [Technical Brief](TECHNICAL_BRIEF.md) | [Decision Engine Plan](DECISION_ENGINE_PLAN.md)*
