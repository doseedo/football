# Marshall AI - Internal Development Plan

This is our working plan, not the pitch to Marshall.

---

# Two-Part Strategy

## Part 1: Tracking & Decision Engine
**What:** Computer vision system to extract player positions and tactical analysis from video
**Timeline:** Months of development
**Value:** Delayed until working reliably

## Part 2: Marshall-Specific AI
**What:** Knowledge system trained on all Marshall program data
**Timeline:** Can start immediately
**Value:** Instant - data already exists

**Key insight:** Part 2 delivers value NOW while Part 1 develops. Part 1 eventually feeds into Part 2 as another data source.

---

# Part 2: Marshall-Specific AI

## Available Data Sources

| Data Source | What It Contains | Available Now? |
|-------------|------------------|----------------|
| Wyscout | Match events, player stats, opponent tendencies, video | Yes |
| GPS (Catapult/other) | Training load, distances, sprints, accelerations | Yes |
| Financials | Budget, scholarships, travel, staff costs | Yes |
| Game Plans | Tactical documents, formation notes, opponent scouting | Yes |
| Meeting Recordings | Decisions, context, discussions (transcribed) | Yes |
| NCAA Rulebook | Compliance rules, deadlines, limits | Yes (public) |
| Recruitment | Prospect notes, evaluations, transfer portal | Yes |
| Historical Rosters | Past players, development, outcomes | Yes |
| Schedule/Calendar | Matches, travel, recovery windows | Yes |
| Medical/Injury | Injury history, return-to-play | Yes |

## What We Build

### Core System: RAG (Retrieval-Augmented Generation)

Not a fine-tuned model. A system that:
1. Ingests and indexes all Marshall data
2. Retrieves relevant information based on queries
3. Uses LLM to synthesize answers with citations

**Why RAG over fine-tuning:**
- Works with data you have now
- Updates as new data comes in
- Transparent (shows sources)
- No expensive training runs

### Technical Components

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│         (Chat / Dashboard / Mobile App)              │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  Query Engine                        │
│     (Understands question, routes to data)          │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Retrieval Layer                         │
│   (Vector search + structured queries + filters)    │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 Data Layer                           │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Wyscout │ │   GPS    │ │Financial │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Game Plans│ │ Meetings │ │   NCAA   │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │Recruitment│ │ Medical │ │ Schedule │            │
│  └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────┘
```

## Why This Is Valuable (Not Just Another Tool)

**The point is NOT to replace Wyscout.** You can already query Wyscout.

**The point is connecting data that's currently siloed.** Things no single tool can do.

### Cross-Source Pattern Recognition

| Question | Why Single Tools Can't Answer |
|----------|------------------------------|
| "When we play high-press teams, do our players fatigue more?" | Needs Wyscout opponent style + GPS load + results |
| "What training week structure preceded our best performances?" | Needs GPS + schedule + match results |
| "Which of OUR players performs best against pressing teams?" | Needs Wyscout opponent data + our performance data |
| "What load patterns preceded injuries?" | Needs GPS + medical history |
| "Which recruits are similar to players who succeeded here?" | Needs recruitment profiles + historical roster outcomes |

### Institutional Memory (Stuff That Gets Lost)

- "What did we learn recruiting Player X that applies to similar prospects?"
- "Why did we change the pressing trigger against Team Y last year?"
- "What concerns did we have about this player before signing? Were they valid?"
- "What's our historical record against 4-4-2 teams and what worked?"

This knowledge lives in people's heads. When staff turn over, it's gone.

### Preparation That Writes Itself

**Before a match, auto-generate:**
- Opponent scouting brief (Wyscout data)
- What we said about them last time (meeting transcripts)
- Physical state of our squad (GPS + medical)
- Historical results and what worked (past game plans + outcomes)
- Compliance check (training hours vs NCAA limit)

One document. Currently this is manual assembly from 5 sources.

### Proactive Alerts (System Tells You)

- "Player X's load pattern matches profile before his last injury"
- "Opponent Y changed their corner routine since last scouting"
- "Training hours at 85% of NCAA weekly limit with 2 days left"
- "Budget tracking 20% over projection for travel"
- "Recruit Z just entered transfer portal - matches your fullback criteria"

You don't have to ask. It watches and flags.

### Meeting Intelligence

**Searchable decisions:**
- "What did Coach say about Player X's positioning in September?"
- "When did we decide to change the set piece routine?"
- "What were the concerns about Recruit Y?"

**Decision → Outcome tracking:**
- "We decided X. Did it work?"
- Pattern recognition: which types of decisions led to good outcomes?

**Staff alignment:**
- New staff can search past context
- "What's our philosophy on X?" returns actual discussions, not just documents

### Recruitment Intelligence

- "Find prospects whose profile matches [player who succeeded here]"
- "What's our conversion rate from ID camps to signed players?"
- "Which recruiting channels produced our best players?"
- "What did we say about Recruit Z six months ago vs now?"

### Game Model Accountability

- "How often did we actually achieve field tilt this season?"
- "What's our counter-press success rate?"
- "Do we play the way we say we want to play?"

Initially from Wyscout event data. Later from tracking when Part 1 is ready.

---

## Meeting Recording & Video Capture Setup

### The Problem
Meetings often show content on TV (Wyscout clips, presentations, tactical boards). Audio transcription alone misses visual context.

### Recommended Setup

**Minimum (Free):**
```
Laptop → Screen record with OBS/Loom (free)
      → Present to TV
Phone  → Record audio for transcription
```

**Better ($30):**
```
Laptop → HDMI Capture Dongle ($30) → Records everything shown
      → TV (passthrough, displays normally)
Phone  → Backup audio + room context
```

**Best ($100-150):**
```
Laptop → Elgato Cam Link 4K ($100) → High quality capture
      → TV (passthrough)
Camera → Room view (whiteboard, gestures)
```

### Equipment Options

| Device | Cost | Use Case |
|--------|------|----------|
| **OBS Studio** | Free | Screen recording from laptop |
| **Loom** | Free | Easy screen recording, auto-uploads |
| **Generic HDMI capture** | $20-40 | Captures TV feed, good enough quality |
| **Elgato Cam Link 4K** | ~$100 | Higher quality capture |
| **Phone on tripod** | ~$20 tripod | Room audio + whiteboard capture |

### Transcription Options

| Option | Cost | Notes |
|--------|------|-------|
| **Whisper (OpenAI)** | Free | Open source, excellent accuracy, self-hosted |
| **Otter.ai** | Free: 300 min/month | Auto-transcription, searchable |
| **Otter.ai Pro** | ~$17/user/month | More minutes, better features |
| **Fireflies.ai** | Free tier / ~$19/month | Auto-joins meetings |

**Recommendation:** Start with Whisper (free) for transcription. Upgrade to Otter if you want live transcription and auto-join features.

### What Gets Captured

| Content Type | Capture Method |
|--------------|----------------|
| Wyscout video clips | Screen record / HDMI capture |
| Tactical presentations | Screen record / HDMI capture |
| Whiteboard drawings | Camera on tripod |
| Discussion audio | Phone + Whisper transcription |
| Drawing on tablet | Screen record |

### Processing Pipeline

```
Meeting Recording
       ↓
┌──────┴──────┐
│             │
Video      Audio
│             │
Index      Whisper
frames    transcribe
│             │
└──────┬──────┘
       ↓
  Linked & Searchable
       ↓
"Show me when we discussed Team X's press"
       ↓
Returns: video clip + transcript + timestamp
```

### Storage Requirements

| Content | Size | Notes |
|---------|------|-------|
| Meeting audio (1 hr) | ~60 MB | Compressed |
| Meeting video (1 hr) | ~500 MB - 1 GB | Screen capture |
| Transcript | ~10 KB | Text |
| **Per week estimate** | ~1-2 GB | If recording 1-2 meetings |
| **Per year** | ~50-100 GB | Manageable |

---

## Example Queries (What It Can Do)

**Opponent Prep:**
- "What are Team X's corner kick tendencies?"
- "How does Team X play against a high press?"
- "Show me when we discussed Team X in meetings this season"

**Performance:**
- "Which players have highest training load this week?"
- "Who's at elevated injury risk based on load + history?"
- "Compare Player A's sprint numbers to last month"

**Compliance:**
- "How many training hours have we logged this week?"
- "When does the contact period end?"
- "Are we compliant on scholarship limits?"

**Recruitment:**
- "List prospects we've scouted at left back"
- "What did we say about Recruit X in the last meeting?"
- "Show me transfer portal players that fit our system"

**Institutional Knowledge:**
- "What did we decide about formation against 4-3-3 teams?"
- "What was the scouting report on Player X before we signed them?"
- "What budget do we have remaining for spring travel?"
- "Play the clip where we discussed the pressing adjustment"

## Build Sequence

### Phase 2.1: Data Ingestion (Weeks 1-3)
- Connect to Wyscout API (or export data)
- Structure GPS data exports
- Index game plans and documents
- Transcribe meeting recordings
- Parse NCAA rulebook

**Deliverable:** All data in queryable format

### Phase 2.2: Core RAG System (Weeks 4-6)
- Set up vector database (Pinecone, Weaviate, or local)
- Build retrieval pipeline
- Connect to LLM (Claude API, GPT-4, or local)
- Basic query interface

**Deliverable:** Can ask questions, get sourced answers

### Phase 2.3: Specialized Modules (Weeks 7-10)
- Opponent scouting assistant
- Training load dashboard
- Compliance checker
- Recruitment tracker

**Deliverable:** Purpose-built tools for common workflows

### Phase 2.4: Integration & Refinement (Weeks 11-12)
- User feedback loop
- Improve retrieval accuracy
- Add more data sources
- Mobile/quick access options

**Deliverable:** System staff actually use

---

# Part 1: Tracking & Decision Engine

## What We Have

| Component | Status |
|-----------|--------|
| Player detection | Working |
| Ball detection | Working |
| Manual pitch calibration | Working |
| Automatic pitch calibration | Code exists, needs training |
| Basic tracking | Working (loses players in occlusions) |
| Physical metrics | Working |
| Decision engine code | Exists, needs validated tracking |

## What We Need to Build

### Phase 1.1: Automatic Calibration (Weeks 1-6)
- Label Marshall pitch footage for training
- Train HRNet keypoint detector
- Validate on different camera angles

**Deliverable:** Process video without manual point clicking

### Phase 1.2: Tracking Improvements (Weeks 7-12)
- Improve occlusion handling
- Integrate off-screen extrapolation
- Test on full matches

**Deliverable:** More reliable player tracking

### Phase 1.3: Validation (Weeks 13-16)
- Compare to GPS ground truth
- Validate physical metrics accuracy
- Test decision engine outputs with coaches

**Deliverable:** Trusted data that can feed into Part 2

### Phase 1.4: Integration (Weeks 17-20)
- Connect tracking output to Part 2 data layer
- Match data becomes queryable
- "Show me our pressing intensity vs Team X" becomes possible

**Deliverable:** Tracking feeds the Marshall AI

---

# How They Merge

```
Timeline:
─────────────────────────────────────────────────────────────────

Part 2 (Marshall AI):
[===Data Ingestion===][===Core RAG===][===Modules===][===Refine===]
     Weeks 1-3           Weeks 4-6     Weeks 7-10    Weeks 11-12
                              │
                              ▼
                    Immediate Value Starts

Part 1 (Tracking):
[=====Calibration=====][====Tracking====][==Validation==][Integration]
      Weeks 1-6           Weeks 7-12       Weeks 13-16    Weeks 17-20
                                                              │
                                                              ▼
                                              Tracking feeds into Part 2

─────────────────────────────────────────────────────────────────

Week 6:  Part 2 answering questions about Wyscout, GPS, game plans
Week 12: Part 2 fully operational with all existing data
Week 16: Part 1 tracking validated
Week 20: Tracking data flows into Part 2, richer analysis possible
```

---

# Resource Requirements

## Part 2 (Marshall AI)
- Developer time: 1 person, 10-12 weeks
- LLM API costs: ~$100-500/month depending on usage
- Vector database: Free tier or ~$50-100/month
- Infrastructure: Minimal (can run on basic cloud)

## Part 1 (Tracking)
- Developer time: 1 person, 16-20 weeks
- GPU compute for training: ~$500-1000
- Test footage: Marshall practice/match video

## Ongoing
- LLM costs scale with usage
- Maintenance and improvements
- New data source integration

---

# Detailed Cost Breakdown

## One-Time Setup Costs

| Item | Cost | Notes |
|------|------|-------|
| HDMI capture dongle | $30-100 | For meeting recording |
| Phone tripod | $20 | Room capture |
| **Total setup** | **$50-120** | |

## Monthly Operating Costs

| Component | Low Usage | Moderate Usage | Heavy Usage |
|-----------|-----------|----------------|-------------|
| Data storage | $0-5 | $5-10 | $10-20 |
| Meeting transcription | $0 (Whisper) | $17 (Otter Pro) | $17-40 |
| Vector database | $0 (free tier) | $25 | $70 |
| LLM API (Claude/GPT) | $50 | $100-150 | $200-300 |
| **Monthly total** | **$50-75** | **$150-200** | **$300-430** |

## What Drives Cost

- **LLM API is the main cost** - scales with query volume
- **Storage is cheap** - text and transcripts are small
- **Free tiers cover a lot** - Pinecone, Otter, university storage

## Cost-Saving Options

1. Use university Google Drive / OneDrive (often unlimited for .edu)
2. Use Whisper locally instead of paid transcription
3. Use free vector DB tier (covers ~100K documents)
4. Batch queries to reduce LLM calls
5. Cache common queries

---

# Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data quality issues | Bad answers | Validate sources, show citations |
| LLM hallucination | Wrong information | RAG grounds answers in real data |
| Tracking doesn't improve | Part 1 delayed | Part 2 works without it |
| Staff don't adopt | Wasted effort | Start with highest-value queries |
| Data access problems | Can't ingest | Work with Marshall IT early |

---

# Success Metrics

## Part 2
- Staff using system weekly
- Questions answered accurately (spot check)
- Time saved on opponent prep
- Compliance issues caught

## Part 1
- Calibration works on new footage without manual work
- Tracking accuracy vs GPS baseline
- Coach validates decision engine outputs make sense

---

# Next Steps

1. **Confirm data access** - What can we actually get from Marshall?
2. **Prioritize queries** - What questions matter most to staff?
3. **Set up data pipeline** - Start ingesting what's available
4. **Build MVP** - Simplest useful version of Part 2
5. **Parallel Part 1** - Begin calibration training with footage

---

# Converting This to Pitch

When we turn this into the Marshall pitch:
- Lead with Part 2 (immediate value)
- Part 1 is "background development that makes it better over time"
- Focus on what they can DO, not how it works
- Show example queries relevant to their workflows
- De-emphasize technical details
- Emphasize: "use data you already have"
