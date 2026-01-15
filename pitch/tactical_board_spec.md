# Tactical Board
## Decision Engine Visualization Interface

---

## Concept

The Tactical Board is an interactive visualization layer that displays the real-time state of the Decision Engine. It transforms raw tracking coordinates into an intuitive coaching interface.

**Core Idea:** Every tracked game feeds into a unified tactical board that shows not just where players are, but what the Decision Engine understands about the tactical situation.

---

## Interface Design

### Main View

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TACTICAL BOARD                    Marshall vs Kentucky │ 34:21 │ ▶ LIVE │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────┐  ┌────────────┐ │
│  │                                                    │  │   STATE    │ │
│  │              ○ 9        ● 10                       │  │            │ │
│  │         ○ 11      ● 8         ● 7    ○ 7          │  │ ███████░░░ │ │
│  │                                                    │  │ ATTACKING  │ │
│  │      ○ 6       ◉        ● 6           ○ 11        │  │ 78% conf   │ │
│  │                (ball)                              │  │            │ │
│  │  ○ 3      ● 4      ● 5      ● 2         ○ 3       │  ├────────────┤ │
│  │                                                    │  │ POSSESSION │ │
│  │         ○ 4    ● 3                ○ 2             │  │ Marshall   │ │
│  │              ○ 5                                   │  │ ████████░░ │ │
│  │                    ○ 1                            │  │    67%     │ │
│  │                                                    │  │            │ │
│  └────────────────────────────────────────────────────┘  ├────────────┤ │
│                                                          │  METRICS   │ │
│  ════════════════════════●════════════════════════════   │            │ │
│  0:00                   34:21                    90:00   │ Line: 44m  │ │
│                                                          │ Width: 38m │ │
│  [◀◀] [◀] [▶ PLAY] [▶▶] [⏸]     Speed: [1x ▼]          │ Compact:26m│ │
│                                                          │ Press: 0.72│ │
│  LAYERS: [●] Players [●] Ball [○] Voronoi [○] Passing   │ Space: 58% │ │
│          [○] Ghosts  [○] Pressure [●] Trails            └────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Components

**1. Pitch View (Center)**
- 2D pitch with accurate dimensions
- Player positions from tracking data
- Ball position with trail
- Color-coded by team (customizable)
- Jersey numbers displayed

**2. State Panel (Right)**
- Current tactical state (ATTACKING, DEFENDING, etc.)
- Confidence score
- Possession indicator
- Key metrics dashboard

**3. Timeline (Bottom)**
- Scrubbing through match
- Play/pause controls
- Playback speed control
- Bookmark/clip functionality

**4. Layer Toggles**
- Turn visualization layers on/off
- Customize what's displayed

---

## Visualization Layers

### Layer 1: Players & Ball (Default)
Basic view showing all player positions and ball.

```
    ○ 9 ────trail────→ ○ 9

    ● = Home team
    ○ = Away team
    ◉ = Ball
```

### Layer 2: Voronoi Space Control
Shows which team controls which areas of the pitch.

```
┌─────────────────────────────────┐
│░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│░░░░░ ○ ░░░░│▓▓▓▓ ● ▓▓▓▓▓▓▓▓▓▓▓│
│░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
└─────────────────────────────────┘

░ = Away team space (42%)
▓ = Home team space (58%)
```

### Layer 3: Passing Lanes
Shows available passing options from ball carrier.

```
         ○ target
        ╱
    ● ══════════ ○ blocked
   ball    ╲
            ○ target

═══ = Open lane
─── = Contested lane
╳╳╳ = Blocked lane
```

### Layer 4: Pressure Zones
Heat map showing pressing intensity.

```
┌─────────────────────────────────┐
│                    ▓▓▓▓▓▓▓▓▓▓▓▓│
│                  ▓▓████████▓▓▓▓│
│                ▓▓██████████████│ ← High pressure zone
│                  ▓▓████████▓▓▓▓│
│                    ▓▓▓▓▓▓▓▓▓▓▓▓│
│                                 │
└─────────────────────────────────┘
```

### Layer 5: Ghost Positions (Decision Engine)
Shows where players SHOULD be vs where they ARE.

```
    ○ actual position
     ╲
      ◌ ← ghost (optimal position)

    Arrow shows required movement
    Color indicates urgency (green/yellow/red)
```

### Layer 6: Movement Trails
Shows recent player movement paths.

```
    ·····○ 9
         │
    Path shows last 3-5 seconds of movement
```

---

## Modes

### 1. Review Mode
Scrub through completed matches, analyze specific moments.

**Features:**
- Full timeline control
- Clip creation (save 10-30 second segments)
- Side-by-side comparison (two moments)
- Annotation tools (draw on pitch)
- Export to video

### 2. Live Mode
Real-time display during matches (if processing live feed).

**Features:**
- ~2-3 second delay from live action
- Auto-updating metrics
- Alert system for tactical triggers
- Quick-clip last 30 seconds

### 3. Simulation Mode
"What-if" scenario testing.

**Features:**
- Drag players to new positions
- Model shows predicted outcomes
- Compare formations
- Test pressing triggers

### 4. Training Mode
Film session interface for player development.

**Features:**
- Focus on individual player
- Show optimal position overlay
- Loop specific sequences
- Compare to benchmark players

---

## Data Architecture

### Input: Tracked Match Data

```json
{
  "frame": 51234,
  "timestamp": 34.21,
  "tactical_state": {
    "state": "ATTACKING",
    "confidence": 0.78
  },
  "players": [
    {
      "id": 1,
      "team": "home",
      "jersey": 10,
      "position": {"x": 45.2, "y": 32.1},
      "velocity": {"vx": 3.2, "vy": -1.1},
      "optimal_position": {"x": 47.8, "y": 30.5},
      "deviation": 3.1
    }
  ],
  "ball": {"x": 42.1, "y": 28.3, "z": 0.4},
  "metrics": {
    "possession": 0.67,
    "home_space_control": 0.58,
    "defensive_line": 44.2,
    "pressing_intensity": 0.72
  }
}
```

### Storage: Match Database

All tracked games stored in queryable database:

```
matches/
├── 2024-10-15_marshall_vs_kentucky/
│   ├── tracking.json       # Frame-by-frame positions
│   ├── tactical.json       # Decision engine states
│   ├── metrics.json        # Aggregated metrics
│   └── video.mp4           # Source video (synced)
├── 2024-10-08_marshall_vs_wku/
│   └── ...
└── index.json              # Match metadata
```

### Query Examples

**"Show me all moments where we lost defensive shape":**
```sql
SELECT timestamp, frame
FROM tactical
WHERE state = 'TRANSITION_DEFENSE'
  AND defensive_line < 35
  AND compactness > 40
```

**"Find pressing triggers where #8 was late":**
```sql
SELECT timestamp, player_8.deviation
FROM players
WHERE tactical_state = 'PRESSING'
  AND player_8.deviation > 5
```

---

## Use Cases

### 1. Post-Match Review (Staff)

**Workflow:**
1. Load match into Tactical Board
2. Jump to key moments (goals, chances, set pieces)
3. Enable Voronoi layer — see space control
4. Enable Ghost layer — see positioning errors
5. Clip sequences for team presentation

**Output:** 5-10 clips with tactical annotations for film session

### 2. Individual Player Review

**Workflow:**
1. Filter to show only target player + ball
2. Enable Ghost layer (optimal positions)
3. Scrub through match, identify deviations
4. Clip examples of good/bad positioning
5. Export player development report

**Output:** "Here's 3 moments you were out of position, here's what it should have looked like"

### 3. Opponent Scouting

**Workflow:**
1. Load opponent's recent match
2. Enable Pressure Zones — see their pressing patterns
3. Enable Passing Lanes — see their build-up routes
4. Identify vulnerabilities
5. Export scouting clips

**Output:** Pre-match tactical briefing with visual evidence

### 4. Live Match Monitoring

**Workflow:**
1. Connect to live video feed
2. Board updates in real-time (~2s delay)
3. Monitor fatigue metrics, defensive shape
4. Alert when pressing intensity drops below threshold
5. Inform substitution decisions

**Output:** Real-time tactical intelligence for technical staff

### 5. Tactical Experimentation (Simulation)

**Workflow:**
1. Load moment from past match
2. Pause and enter Simulation Mode
3. Drag player to different position
4. Model predicts: space control change, passing lane impact
5. Test alternative formations

**Output:** "If we played a back 3 here, we'd have 12% more space control"

---

## Technical Implementation

### Frontend Options

| Option | Pros | Cons |
|--------|------|------|
| **Web (React)** | Cross-platform, easy updates, shareable | Needs internet |
| **Electron** | Desktop app, offline capable | Separate install |
| **iPad (Swift)** | Touch-native, portable | iOS only, separate codebase |

**Recommendation:** Web-first (React + Canvas/WebGL), with offline caching for film rooms.

### Key Libraries

```
Rendering:     Pixi.js or Three.js (2D/3D pitch rendering)
State:         React + Zustand (UI state)
Data:          IndexedDB (local cache) + SQLite (queries)
Video Sync:    Video.js with frame-accurate sync
Export:        FFmpeg.wasm (client-side clip export)
```

### Performance Targets

| Metric | Target |
|--------|--------|
| Frame rate | 60 FPS playback |
| Scrub latency | <100ms to new frame |
| Load time | <3s for full match |
| Memory | <500MB for 90min match |

---

## Development Phases

### Phase 1: Core Board (MVP)
- 2D pitch with player dots
- Timeline scrubbing
- Basic metrics panel
- Single match loading

### Phase 2: Visualization Layers
- Voronoi space control
- Passing lanes
- Movement trails
- Ghost positions

### Phase 3: Interaction
- Clip creation/export
- Annotation tools
- Side-by-side comparison
- Player filtering

### Phase 4: Advanced Features
- Live mode connection
- Simulation mode
- Multi-match queries
- Team/opponent library

---

## Integration with Demo/Pilot

### Demo Enhancement
The Tactical Board becomes the centerpiece of the demo:
- More visual than showing JSON/CSV data
- Coaches immediately understand the interface
- "This is what you'd see every week"

### Pilot Deliverable
Instead of just reports, pilot includes:
- Access to Tactical Board with their 5 matches loaded
- Training session on using the interface
- Clip export for their own film sessions

---

## Competitive Advantage

No other affordable solution offers this combination:
- **Hudl Sportscode:** Manual tagging, no AI positioning
- **Wyscout:** Video only, no tracking
- **SkillCorner:** Provides data, not an interface
- **This:** Tracking + AI + Interface in one package

**For Marshall:** "You're not just getting data. You're getting a tactical command center."
