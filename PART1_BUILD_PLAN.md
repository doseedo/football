# Part 1 Build Plan - Marshall-Specific AI

Internal development plan for building the Marshall AI system.

---

# Overview

**Goal:** Build a system that connects Marshall's data (Wyscout, GPS, meetings, recruitment, game plans) and makes it queryable.

**Timeline:** 6 weeks to working system

**Five value areas to deliver:**
1. Cross-Source Pattern Recognition
2. Institutional Memory
3. Recruitment Intelligence
4. Proactive Alerts
5. Auto-Generated Prep

---

# Technical Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│              (Web app / Chat interface)              │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  Query Engine                        │
│     (Routes queries, combines results, LLM layer)   │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Vector Database + Search                │
│         (Embeddings for semantic search)            │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 Data Layer                           │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Wyscout │ │   GPS    │ │Game Plans│            │
│  └──────────┘ └──────────┘ └──────────┘            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Meetings │ │Recruitment│ │ Medical │            │
│  └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────┘
```

---

# Data Ingestion Plan

## 1. Wyscout Data

**What we need:**
- Match events (goals, shots, passes, etc.)
- Player statistics
- Opponent tendencies
- Team statistics

**How to get it:**
- Option A: Wyscout API (if Marshall has API access)
- Option B: Manual exports (CSV/Excel) from Wyscout platform
- Option C: Screen scraping as last resort

**Data structure:**
- Match-level data (events, lineups, scores)
- Player-level aggregates (season stats)
- Team-level tendencies (formation, style metrics)

**Questions for Marshall:**
- Do you have Wyscout API access?
- What export formats are available?
- How often is data updated?

---

## 2. GPS Data

**What we need:**
- Training session data (load, distance, sprints)
- Match physical data
- Player-level metrics over time

**How to get it:**
- Export from GPS provider (Catapult, STATSports, etc.)
- Usually CSV or API available

**Data structure:**
- Session date, player, duration
- Metrics: total distance, high-speed running, sprint count, accelerations
- Training vs match flag

**Questions for Marshall:**
- What GPS system do you use?
- What export options exist?
- How far back does data go?

---

## 3. Meeting Recordings

**What we need:**
- Audio recordings → transcripts
- Video of screen content (optional but valuable)
- Timestamps linked to topics discussed

**How to capture:**
- Screen recording (OBS/Loom) from presenting laptop
- Audio recording (phone or dedicated mic)
- HDMI capture dongle for TV content

**Processing pipeline:**
```
Audio file → Whisper transcription → Text with timestamps
Video file → Frame extraction → Key moments indexed
Both → Linked and searchable
```

**Storage:**
- Audio: ~60 MB/hour
- Video: ~500 MB - 1 GB/hour
- Transcripts: ~10 KB/hour

**Questions for Marshall:**
- How many meetings per week?
- What's typically shown on screen?
- Who handles recording logistics?

---

## 4. Game Plans & Documents

**What we need:**
- Scouting reports
- Tactical documents
- Formation notes
- Set piece plans

**How to get it:**
- Collect existing documents (PDFs, Word, etc.)
- Ongoing: save new documents to shared folder

**Processing:**
- PDF/Word → text extraction
- Index by opponent, date, topic
- Link to related meetings and matches

---

## 5. Recruitment Data

**What we need:**
- Prospect profiles
- Evaluation notes
- Communication history
- Outcomes (signed, passed, etc.)

**How to get it:**
- Export from recruiting software (if used)
- Consolidate from spreadsheets
- Connect to meeting discussions about recruits

**Data structure:**
- Prospect: name, position, school, graduation year
- Evaluations: dates, notes, ratings
- Status: pipeline stage, decision, outcome

---

## 6. Medical/Injury Data

**What we need:**
- Injury history
- Return-to-play dates
- Any load restrictions

**Sensitivity:** This data requires careful handling. May need to limit access or anonymize.

---

## 7. Schedule Data

**What we need:**
- Match schedule
- Training schedule
- Travel dates

**How to get it:**
- Calendar export
- Manual entry
- Sync from scheduling system

---

# Week-by-Week Build Plan

## Week 1: Setup & Data Collection

**Goals:**
- Set up development environment
- Get access to all data sources
- Start collecting meeting recordings

**Tasks:**
- [ ] Set up cloud infrastructure (or local dev environment)
- [ ] Get Wyscout export/API access
- [ ] Get GPS data export
- [ ] Collect existing game plans and documents
- [ ] Set up meeting recording workflow
- [ ] Get recruitment data export
- [ ] Create data storage structure

**Deliverable:** All data sources identified and initial exports collected

---

## Week 2: Data Processing & Indexing

**Goals:**
- Process all collected data
- Build initial vector database
- Test basic search

**Tasks:**
- [ ] Parse Wyscout data into structured format
- [ ] Parse GPS data into structured format
- [ ] Process meeting transcripts (if any existing recordings)
- [ ] Extract text from game plan documents
- [ ] Structure recruitment data
- [ ] Set up vector database (Pinecone/Chroma/Weaviate)
- [ ] Create embeddings for all text content
- [ ] Test basic semantic search

**Deliverable:** All data indexed and searchable

---

## Week 3: Core Query System

**Goals:**
- Build query routing logic
- Connect LLM for response generation
- Basic chat interface

**Tasks:**
- [ ] Set up LLM connection (Claude API)
- [ ] Build RAG pipeline (retrieve → augment → generate)
- [ ] Create query classification (what type of question?)
- [ ] Build response formatting
- [ ] Create simple web interface or CLI
- [ ] Test with sample queries

**Deliverable:** Can ask questions and get answers with sources

---

## Week 4: Cross-Source Connections

**Goals:**
- Link data across sources
- Enable pattern queries
- Refine retrieval

**Tasks:**
- [ ] Link meetings to matches (by date, opponent mentioned)
- [ ] Link GPS data to matches
- [ ] Link recruitment discussions to prospect profiles
- [ ] Build cross-source query handling
- [ ] Test pattern recognition queries
- [ ] Refine retrieval accuracy

**Deliverable:** Cross-source queries working

---

## Week 5: Alerts & Auto-Prep

**Goals:**
- Build proactive alert system
- Build auto-generated prep documents
- Scheduled jobs

**Tasks:**
- [ ] Define alert triggers (load thresholds, opponent changes, portal activity)
- [ ] Build alert checking logic
- [ ] Create notification system (email/Slack/app)
- [ ] Build prep document template
- [ ] Create auto-prep generation for upcoming matches
- [ ] Schedule regular data refreshes

**Deliverable:** Alerts firing, prep docs generating

---

## Week 6: Refinement & Handoff

**Goals:**
- Fix issues found in testing
- Improve accuracy
- Document and hand off

**Tasks:**
- [ ] Staff testing and feedback
- [ ] Fix retrieval issues
- [ ] Improve response quality
- [ ] Add missing data connections
- [ ] Document system for ongoing use
- [ ] Train staff on usage
- [ ] Set up ongoing data ingestion workflow

**Deliverable:** Working system in use by staff

---

# Technical Decisions

## Vector Database Choice

| Option | Pros | Cons |
|--------|------|------|
| **Pinecone** | Managed, scalable, free tier | Cloud dependency |
| **Chroma** | Local, free, simple | Less scalable |
| **Weaviate** | Hybrid search, self-host option | More complex |

**Recommendation:** Start with Chroma (local, free) for development, migrate to Pinecone if needed for scale.

---

## LLM Choice

| Option | Pros | Cons |
|--------|------|------|
| **Claude API** | Strong reasoning, long context | Cost |
| **GPT-4** | Widely used, good quality | Cost |
| **GPT-3.5** | Cheaper | Lower quality |
| **Local (Llama)** | Free, private | Requires GPU, lower quality |

**Recommendation:** Claude API for quality. Can optimize costs with caching and batching.

---

## Hosting

| Option | Pros | Cons |
|--------|------|------|
| **Local server** | Private, no ongoing cost | Maintenance, availability |
| **Cloud (AWS/GCP)** | Scalable, reliable | Monthly cost |
| **Vercel/Railway** | Easy deployment | May have limits |

**Recommendation:** Start local for development, deploy to cloud for reliability.

---

# What We Need From Marshall

## Data Access (Week 1)
- [ ] Wyscout login or API credentials
- [ ] GPS system access/export capability
- [ ] Existing game plans and scouting documents
- [ ] Recruitment tracking data (spreadsheets or system export)
- [ ] Permission to record and store meeting content

## Equipment
- [ ] HDMI capture dongle (~$30) for meeting recording
- [ ] Tripod for phone/camera (~$20)

## Time
- [ ] Someone to handle meeting recordings
- [ ] Periodic feedback on system outputs
- [ ] 1-2 hours/week for testing during development

## Decisions
- [ ] Who has access to the system?
- [ ] What data is too sensitive to include?
- [ ] Preferred interface (web, Slack, mobile)?

---

# Ongoing Operations

After initial build:

**Daily:**
- New meeting recordings processed
- Alerts checked and sent

**Weekly:**
- GPS data refreshed
- Wyscout data updated
- Prep docs generated for upcoming matches

**Monthly:**
- Review system usage
- Identify missing data or features
- Update document index

**Cost estimate:** $50-200/month depending on usage (mainly LLM API)

---

# Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Data quality issues | Validate sources, show citations |
| LLM hallucination | RAG grounds answers in real data |
| Staff don't adopt | Start with highest-value use cases, get feedback early |
| Meeting recording burden | Make it as simple as possible, consider auto-recording |
| Data access delays | Start with whatever is available, add sources incrementally |

---

# Success Metrics

**Week 4 (core working):**
- Can query across all connected data sources
- Retrieval accuracy >80% (relevant results returned)
- Staff can get useful answers

**Week 6 (refined):**
- Alerts firing appropriately
- Prep docs useful for match preparation
- Staff using system regularly

**Month 2+:**
- System part of regular workflow
- New meetings being captured and indexed
- Cross-source insights being discovered

---

# Next Steps

1. **Confirm data access** — What can we get from Marshall this week?
2. **Set up recording** — Get equipment, test workflow
3. **Start development** — Week 1 tasks
4. **Regular check-ins** — Weekly progress review

---

# Questions to Resolve

- What's the Wyscout situation (API vs export)?
- What GPS system is used?
- Where do game plans currently live?
- How is recruitment currently tracked?
- Who will handle meeting recordings?
- Preferred way to access the system (web, Slack, etc.)?
