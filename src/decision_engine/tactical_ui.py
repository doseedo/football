"""
Integrated Tactical UI for Football Decision Engine.

Combines player tracking visualization with decision engine simulation
in a web-based interface.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from .action_executor import DecisionStep, DecisionEngine, create_test_scenario
from .state_scoring import GameState, ActionOption
from .pitch_geometry import Position
from .elimination import Player


def generate_tactical_ui(
    steps: Optional[List[DecisionStep]] = None,
    initial_state: Optional[GameState] = None,
    output_path: str = "tactical_ui.html"
) -> str:
    """
    Generate an integrated tactical UI that combines:
    - Player tracking visualization
    - Decision engine simulation
    - Scenario editor
    - Action analysis
    """

    # Prepare simulation frames if steps provided
    frames = []
    if steps:
        for i, step in enumerate(steps):
            frame = {
                "step": i + 1,
                "ball": {"x": step.state_before.ball_position.x, "y": step.state_before.ball_position.y},
                "carrier": step.state_before.ball_carrier.id if step.state_before.ball_carrier else None,
                "attackers": [
                    {"id": p.id, "x": p.position.x, "y": p.position.y, "speed": p.max_speed}
                    for p in step.state_before.attackers
                ],
                "defenders": [
                    {"id": p.id, "x": p.position.x, "y": p.position.y, "speed": p.max_speed}
                    for p in step.state_before.defenders
                ],
                "action": None,
                "result": None,
                "message": "",
                "available_actions": []
            }

            # Add available actions
            if step.state_before.available_actions:
                frame["available_actions"] = [
                    {
                        "type": a.action_type.value,
                        "target": {"x": a.target.x, "y": a.target.y},
                        "success_prob": a.success_probability,
                        "expected_value": a.expected_value
                    }
                    for a in step.state_before.available_actions
                ]

            if step.action_chosen:
                frame["action"] = {
                    "type": step.action_chosen.action_type.value,
                    "target": {"x": step.action_chosen.target.x, "y": step.action_chosen.target.y},
                    "success_prob": step.action_chosen.success_probability,
                    "expected_value": step.action_chosen.expected_value
                }

            if step.result:
                frame["result"] = step.result.result.value
                frame["message"] = step.result.message

            frames.append(frame)

        # Add final state
        if steps:
            last = steps[-1]
            final_frame = {
                "step": len(steps) + 1,
                "ball": {"x": last.state_after.ball_position.x, "y": last.state_after.ball_position.y},
                "carrier": last.state_after.ball_carrier.id if last.state_after.ball_carrier else None,
                "attackers": [
                    {"id": p.id, "x": p.position.x, "y": p.position.y, "speed": p.max_speed}
                    for p in last.state_after.attackers
                ],
                "defenders": [
                    {"id": p.id, "x": p.position.x, "y": p.position.y, "speed": p.max_speed}
                    for p in last.state_after.defenders
                ],
                "action": None,
                "result": "end",
                "message": "Simulation complete",
                "available_actions": []
            }
            frames.append(final_frame)

    # Prepare initial state for editor if provided
    initial_data = None
    if initial_state:
        initial_data = {
            "ball": {"x": initial_state.ball_position.x, "y": initial_state.ball_position.y},
            "carrier": initial_state.ball_carrier.id if initial_state.ball_carrier else None,
            "attackers": [
                {"id": p.id, "x": p.position.x, "y": p.position.y, "speed": p.max_speed}
                for p in initial_state.attackers
            ],
            "defenders": [
                {"id": p.id, "x": p.position.x, "y": p.position.y, "speed": p.max_speed}
                for p in initial_state.defenders
            ]
        }

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Football Tactical Analysis</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #4ecca3;
            font-size: 28px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #888;
            font-size: 14px;
        }}
        .main-container {{
            display: flex;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .pitch-panel {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .side-panel {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            width: 350px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        #pitch {{
            background: #2d5a27;
            border: 3px solid #fff;
            border-radius: 5px;
            cursor: crosshair;
        }}
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }}
        .tab {{
            flex: 1;
            padding: 10px;
            background: #0f3460;
            border: none;
            border-radius: 5px;
            color: #888;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .tab.active {{
            background: #4ecca3;
            color: #1a1a2e;
            font-weight: bold;
        }}
        .tab:hover:not(.active) {{
            background: #1a4a7a;
            color: #fff;
        }}
        .controls {{
            display: flex;
            gap: 8px;
            margin: 15px 0;
            flex-wrap: wrap;
            justify-content: center;
        }}
        button {{
            background: #4ecca3;
            color: #1a1a2e;
            border: none;
            padding: 10px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 13px;
            transition: all 0.2s;
        }}
        button:hover {{
            background: #3db892;
            transform: translateY(-1px);
        }}
        button:disabled {{
            background: #444;
            color: #888;
            cursor: not-allowed;
            transform: none;
        }}
        button.secondary {{
            background: #0f3460;
            color: #4ecca3;
        }}
        button.secondary:hover {{
            background: #1a4a7a;
        }}
        button.danger {{
            background: #e94560;
            color: #fff;
        }}
        button.danger:hover {{
            background: #d63850;
        }}
        .info-section {{
            margin: 15px 0;
            padding: 12px;
            background: #0f3460;
            border-radius: 8px;
        }}
        .info-section h4 {{
            color: #4ecca3;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
            font-size: 13px;
        }}
        .info-label {{
            color: #888;
        }}
        .info-value {{
            color: #fff;
            font-weight: 500;
        }}
        .action-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        .action-item {{
            padding: 8px 10px;
            margin: 5px 0;
            background: #1a1a2e;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }}
        .action-item:hover {{
            background: #2a2a4e;
        }}
        .action-item.selected {{
            border-left: 3px solid #4ecca3;
        }}
        .action-type {{
            color: #4ecca3;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .action-value {{
            float: right;
            color: #f1c40f;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
            font-size: 12px;
            justify-content: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
        }}
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }}
        .slider-container label {{
            font-size: 12px;
            color: #888;
            min-width: 60px;
        }}
        input[type="range"] {{
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 3px;
            outline: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #4ecca3;
            border-radius: 50%;
            cursor: pointer;
        }}
        .result-success {{ color: #4ecca3; }}
        .result-fail {{ color: #e94560; }}
        .result-goal {{ color: #f1c40f; }}
        .editor-tools {{
            display: flex;
            gap: 5px;
            margin: 10px 0;
            flex-wrap: wrap;
        }}
        .tool-btn {{
            padding: 8px 12px;
            font-size: 12px;
            border-radius: 5px;
        }}
        .tool-btn.active {{
            background: #4ecca3;
            color: #1a1a2e;
        }}
        .player-info {{
            font-size: 12px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .help-text {{
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }}
        .step-counter {{
            font-size: 24px;
            color: #4ecca3;
            text-align: center;
            margin: 10px 0;
        }}
        .progress-bar {{
            height: 4px;
            background: #0f3460;
            border-radius: 2px;
            margin: 10px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: #4ecca3;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Football Tactical Analysis</h1>
        <p class="subtitle">Decision Engine Simulation &amp; Scenario Editor</p>
    </div>

    <div class="main-container">
        <div class="pitch-panel">
            <canvas id="pitch" width="750" height="480"></canvas>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background: #3498db;"></div>
                    <span>Attackers</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #e74c3c;"></div>
                    <span>Defenders</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #f1c40f;"></div>
                    <span>Ball</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: rgba(78, 204, 163, 0.5);"></div>
                    <span>Actions</span>
                </div>
            </div>
        </div>

        <div class="side-panel">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('simulation')">Simulation</button>
                <button class="tab" onclick="switchTab('editor')">Editor</button>
                <button class="tab" onclick="switchTab('analysis')">Analysis</button>
            </div>

            <!-- Simulation Tab -->
            <div id="simulation-tab" class="tab-content">
                <div class="step-counter">
                    Step <span id="stepNum">1</span> / <span id="totalSteps">{len(frames) if frames else 0}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressBar" style="width: 0%"></div>
                </div>

                <div class="controls">
                    <button onclick="reset()">Reset</button>
                    <button onclick="stepBack()" id="backBtn">Back</button>
                    <button onclick="togglePlay()" id="playBtn">Play</button>
                    <button onclick="stepForward()" id="fwdBtn">Next</button>
                </div>

                <div class="slider-container">
                    <label>Speed</label>
                    <input type="range" id="speed" min="200" max="2000" value="1000" onchange="updateSpeed()">
                </div>

                <div class="info-section">
                    <h4>Current State</h4>
                    <div class="info-row">
                        <span class="info-label">Ball Carrier</span>
                        <span class="info-value" id="carrier">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Ball Position</span>
                        <span class="info-value" id="ballPos">-</span>
                    </div>
                </div>

                <div class="info-section">
                    <h4>Action</h4>
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value" id="actionType">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Target</span>
                        <span class="info-value" id="actionTarget">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Success Prob</span>
                        <span class="info-value" id="successProb">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Expected Value</span>
                        <span class="info-value" id="expValue">-</span>
                    </div>
                </div>

                <div class="info-section">
                    <h4>Result</h4>
                    <div class="info-row">
                        <span class="info-label">Outcome</span>
                        <span class="info-value" id="result">-</span>
                    </div>
                    <div id="message" style="font-size: 12px; color: #888; margin-top: 5px;"></div>
                </div>
            </div>

            <!-- Editor Tab -->
            <div id="editor-tab" class="tab-content" style="display: none;">
                <div class="editor-tools">
                    <button class="tool-btn active" onclick="setTool('select')" id="tool-select">Select</button>
                    <button class="tool-btn" onclick="setTool('attacker')" id="tool-attacker">+ Attacker</button>
                    <button class="tool-btn" onclick="setTool('defender')" id="tool-defender">+ Defender</button>
                    <button class="tool-btn" onclick="setTool('ball')" id="tool-ball">+ Ball</button>
                </div>

                <div class="controls">
                    <button onclick="clearPitch()" class="danger">Clear All</button>
                    <button onclick="loadScenario()">Load Default</button>
                    <button onclick="runSimulation()">Run Simulation</button>
                </div>

                <div class="player-info" id="selectedPlayer" style="display: none;">
                    <h4>Selected Player</h4>
                    <div class="info-row">
                        <span class="info-label">ID</span>
                        <span class="info-value" id="selectedId">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Position</span>
                        <span class="info-value" id="selectedPos">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Team</span>
                        <span class="info-value" id="selectedTeam">-</span>
                    </div>
                    <button onclick="deleteSelected()" class="danger" style="margin-top: 10px; width: 100%;">Delete</button>
                </div>

                <div class="help-text">
                    Click on pitch to add players. Drag to move. Right-click to set ball carrier.
                </div>
            </div>

            <!-- Analysis Tab -->
            <div id="analysis-tab" class="tab-content" style="display: none;">
                <div class="info-section">
                    <h4>Available Actions</h4>
                    <div class="action-list" id="actionList">
                        <div style="color: #666; font-size: 12px;">No actions available</div>
                    </div>
                </div>

                <div class="info-section">
                    <h4>Display Options</h4>
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <label style="display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer;">
                            <input type="checkbox" id="showActions" checked onchange="render()"> Show action arrows
                        </label>
                        <label style="display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer;">
                            <input type="checkbox" id="showAllActions" onchange="render()"> Show all possible actions
                        </label>
                        <label style="display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer;">
                            <input type="checkbox" id="showLabels" checked onchange="render()"> Show player labels
                        </label>
                        <label style="display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer;">
                            <input type="checkbox" id="showGrid" onchange="render()"> Show grid overlay
                        </label>
                    </div>
                </div>

                <div class="info-section">
                    <h4>Simulation Stats</h4>
                    <div class="info-row">
                        <span class="info-label">Total Steps</span>
                        <span class="info-value" id="statSteps">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Successful Actions</span>
                        <span class="info-value" id="statSuccess">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Failed Actions</span>
                        <span class="info-value" id="statFail">0</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Goals</span>
                        <span class="info-value" id="statGoals">0</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data
        const frames = {json.dumps(frames)};
        const initialState = {json.dumps(initial_data) if initial_data else 'null'};

        // State
        let currentFrame = 0;
        let playing = false;
        let playInterval = null;
        let speed = 1000;
        let currentTab = 'simulation';
        let currentTool = 'select';
        let selectedPlayer = null;
        let dragging = false;
        let editorState = {{
            attackers: [],
            defenders: [],
            ball: null,
            carrier: null
        }};
        let highlightedAction = null;

        // Canvas
        const canvas = document.getElementById('pitch');
        const ctx = canvas.getContext('2d');

        // Pitch dimensions
        const PITCH_LENGTH = 105;
        const PITCH_WIDTH = 68;
        const HALF_LENGTH = 52.5;
        const HALF_WIDTH = 34;
        const scaleX = canvas.width / PITCH_LENGTH;
        const scaleY = canvas.height / PITCH_WIDTH;

        function pitchToCanvas(x, y) {{
            return {{
                x: (x + HALF_LENGTH) * scaleX,
                y: (HALF_WIDTH - y) * scaleY
            }};
        }}

        function canvasToPitch(cx, cy) {{
            return {{
                x: cx / scaleX - HALF_LENGTH,
                y: HALF_WIDTH - cy / scaleY
            }};
        }}

        function drawPitch() {{
            ctx.fillStyle = '#2d5a27';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = 'rgba(255,255,255,0.5)';
            ctx.lineWidth = 2;

            // Center line
            const center = pitchToCanvas(0, 0);
            ctx.beginPath();
            ctx.moveTo(center.x, 0);
            ctx.lineTo(center.x, canvas.height);
            ctx.stroke();

            // Center circle
            ctx.beginPath();
            ctx.arc(center.x, center.y, 9.15 * scaleX, 0, Math.PI * 2);
            ctx.stroke();

            // Center spot
            ctx.beginPath();
            ctx.arc(center.x, center.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.fill();

            // Penalty areas
            const penaltyLength = 16.5;
            const penaltyWidth = 40.3;
            const goalAreaLength = 5.5;
            const goalAreaWidth = 18.32;

            // Left penalty area
            const leftPenalty = pitchToCanvas(-HALF_LENGTH, penaltyWidth/2);
            ctx.strokeRect(leftPenalty.x, leftPenalty.y, penaltyLength * scaleX, penaltyWidth * scaleY);

            // Left goal area
            const leftGoalArea = pitchToCanvas(-HALF_LENGTH, goalAreaWidth/2);
            ctx.strokeRect(leftGoalArea.x, leftGoalArea.y, goalAreaLength * scaleX, goalAreaWidth * scaleY);

            // Right penalty area
            const rightPenalty = pitchToCanvas(HALF_LENGTH - penaltyLength, penaltyWidth/2);
            ctx.strokeRect(rightPenalty.x, rightPenalty.y, penaltyLength * scaleX, penaltyWidth * scaleY);

            // Right goal area
            const rightGoalArea = pitchToCanvas(HALF_LENGTH - goalAreaLength, goalAreaWidth/2);
            ctx.strokeRect(rightGoalArea.x, rightGoalArea.y, goalAreaLength * scaleX, goalAreaWidth * scaleY);

            // Penalty spots
            const leftSpot = pitchToCanvas(-HALF_LENGTH + 11, 0);
            const rightSpot = pitchToCanvas(HALF_LENGTH - 11, 0);
            ctx.beginPath();
            ctx.arc(leftSpot.x, leftSpot.y, 3, 0, Math.PI * 2);
            ctx.arc(rightSpot.x, rightSpot.y, 3, 0, Math.PI * 2);
            ctx.fill();

            // Goals
            ctx.fillStyle = '#fff';
            const goalWidth = 7.32;
            const leftGoal = pitchToCanvas(-HALF_LENGTH - 2, goalWidth/2);
            ctx.fillRect(leftGoal.x, leftGoal.y, 2 * scaleX, goalWidth * scaleY);

            const rightGoal = pitchToCanvas(HALF_LENGTH, goalWidth/2);
            ctx.fillRect(rightGoal.x, rightGoal.y, 2 * scaleX, goalWidth * scaleY);

            // Grid overlay
            if (document.getElementById('showGrid')?.checked) {{
                ctx.strokeStyle = 'rgba(255,255,255,0.1)';
                ctx.lineWidth = 1;
                for (let x = -50; x <= 50; x += 10) {{
                    const p = pitchToCanvas(x, 0);
                    ctx.beginPath();
                    ctx.moveTo(p.x, 0);
                    ctx.lineTo(p.x, canvas.height);
                    ctx.stroke();
                }}
                for (let y = -30; y <= 30; y += 10) {{
                    const p = pitchToCanvas(0, y);
                    ctx.beginPath();
                    ctx.moveTo(0, p.y);
                    ctx.lineTo(canvas.width, p.y);
                    ctx.stroke();
                }}
            }}
        }}

        function drawPlayer(x, y, color, label, hasBall, isSelected) {{
            const pos = pitchToCanvas(x, y);

            // Selection ring
            if (isSelected) {{
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 15, 0, Math.PI * 2);
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 3;
                ctx.stroke();
            }}

            // Player circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = hasBall ? '#f1c40f' : '#fff';
            ctx.lineWidth = hasBall ? 3 : 1.5;
            ctx.stroke();

            // Label
            if (document.getElementById('showLabels')?.checked !== false) {{
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 10px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, pos.x, pos.y);
            }}
        }}

        function drawBall(x, y) {{
            const pos = pitchToCanvas(x, y);
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 7, 0, Math.PI * 2);
            ctx.fillStyle = '#f1c40f';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }}

        function drawAction(from, to, success, isHighlighted) {{
            const fromPos = pitchToCanvas(from.x, from.y);
            const toPos = pitchToCanvas(to.x, to.y);

            ctx.beginPath();
            ctx.moveTo(fromPos.x, fromPos.y);
            ctx.lineTo(toPos.x, toPos.y);

            if (isHighlighted) {{
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 5;
                ctx.setLineDash([]);
            }} else {{
                ctx.strokeStyle = success === 'success' || success === 'goal'
                    ? 'rgba(46, 204, 113, 0.8)'
                    : success === null ? 'rgba(78, 204, 163, 0.5)' : 'rgba(231, 76, 60, 0.8)';
                ctx.lineWidth = 3;
                ctx.setLineDash([5, 5]);
            }}
            ctx.stroke();
            ctx.setLineDash([]);

            // Arrow head
            const angle = Math.atan2(toPos.y - fromPos.y, toPos.x - fromPos.x);
            ctx.beginPath();
            ctx.moveTo(toPos.x, toPos.y);
            ctx.lineTo(toPos.x - 12 * Math.cos(angle - 0.4), toPos.y - 12 * Math.sin(angle - 0.4));
            ctx.lineTo(toPos.x - 12 * Math.cos(angle + 0.4), toPos.y - 12 * Math.sin(angle + 0.4));
            ctx.closePath();
            ctx.fillStyle = isHighlighted ? '#fff' : (success === 'success' || success === 'goal'
                ? 'rgba(46, 204, 113, 0.8)'
                : success === null ? 'rgba(78, 204, 163, 0.5)' : 'rgba(231, 76, 60, 0.8)');
            ctx.fill();
        }}

        function render() {{
            drawPitch();

            if (currentTab === 'editor') {{
                renderEditor();
                return;
            }}

            if (frames.length === 0) return;

            const frame = frames[currentFrame];

            // Draw all available actions if enabled
            if (document.getElementById('showAllActions')?.checked && frame.available_actions) {{
                frame.available_actions.forEach((action, i) => {{
                    if (action !== highlightedAction) {{
                        drawAction(frame.ball, action.target, null, false);
                    }}
                }});
            }}

            // Draw highlighted action
            if (highlightedAction) {{
                drawAction(frame.ball, highlightedAction.target, null, true);
            }}

            // Draw chosen action
            if (document.getElementById('showActions')?.checked !== false && frame.action && !highlightedAction) {{
                drawAction(frame.ball, frame.action.target, frame.result, false);
            }}

            // Draw defenders
            frame.defenders.forEach(d => {{
                drawPlayer(d.x, d.y, '#e74c3c', d.id.replace('D_', ''), false, false);
            }});

            // Draw attackers
            frame.attackers.forEach(a => {{
                const hasBall = a.id === frame.carrier;
                drawPlayer(a.x, a.y, '#3498db', a.id, hasBall, false);
            }});

            // Draw ball
            drawBall(frame.ball.x, frame.ball.y);

            updateUI(frame);
        }}

        function renderEditor() {{
            // Draw editor state
            editorState.defenders.forEach((d, i) => {{
                const isSelected = selectedPlayer && selectedPlayer.type === 'defender' && selectedPlayer.index === i;
                const hasBall = editorState.carrier && editorState.carrier.type === 'defender' && editorState.carrier.index === i;
                drawPlayer(d.x, d.y, '#e74c3c', 'D' + (i+1), hasBall, isSelected);
            }});

            editorState.attackers.forEach((a, i) => {{
                const isSelected = selectedPlayer && selectedPlayer.type === 'attacker' && selectedPlayer.index === i;
                const hasBall = editorState.carrier && editorState.carrier.type === 'attacker' && editorState.carrier.index === i;
                drawPlayer(a.x, a.y, '#3498db', 'A' + (i+1), hasBall, isSelected);
            }});

            if (editorState.ball) {{
                drawBall(editorState.ball.x, editorState.ball.y);
            }}
        }}

        function updateUI(frame) {{
            document.getElementById('stepNum').textContent = frame.step;
            document.getElementById('carrier').textContent = frame.carrier || '-';
            document.getElementById('ballPos').textContent =
                `(${{frame.ball.x.toFixed(1)}}, ${{frame.ball.y.toFixed(1)}})`;

            // Progress bar
            const progress = (currentFrame / Math.max(1, frames.length - 1)) * 100;
            document.getElementById('progressBar').style.width = progress + '%';

            if (frame.action) {{
                document.getElementById('actionType').textContent = frame.action.type.toUpperCase();
                document.getElementById('actionTarget').textContent =
                    `(${{frame.action.target.x.toFixed(1)}}, ${{frame.action.target.y.toFixed(1)}})`;
                document.getElementById('successProb').textContent =
                    (frame.action.success_prob * 100).toFixed(0) + '%';
                document.getElementById('expValue').textContent =
                    frame.action.expected_value.toFixed(3);
            }} else {{
                document.getElementById('actionType').textContent = '-';
                document.getElementById('actionTarget').textContent = '-';
                document.getElementById('successProb').textContent = '-';
                document.getElementById('expValue').textContent = '-';
            }}

            const resultEl = document.getElementById('result');
            resultEl.textContent = frame.result || '-';
            resultEl.className = frame.result === 'success' ? 'result-success' :
                                frame.result === 'goal' ? 'result-goal' : 'result-fail';
            document.getElementById('message').textContent = frame.message;

            document.getElementById('backBtn').disabled = currentFrame === 0;
            document.getElementById('fwdBtn').disabled = currentFrame >= frames.length - 1;

            // Update action list
            updateActionList(frame);
        }}

        function updateActionList(frame) {{
            const list = document.getElementById('actionList');
            if (!frame.available_actions || frame.available_actions.length === 0) {{
                list.innerHTML = '<div style="color: #666; font-size: 12px;">No actions available</div>';
                return;
            }}

            list.innerHTML = frame.available_actions.map((action, i) => `
                <div class="action-item ${{action === highlightedAction ? 'selected' : ''}}"
                     onmouseenter="highlightAction(${{i}})"
                     onmouseleave="clearHighlight()">
                    <span class="action-type">${{action.type}}</span>
                    <span class="action-value">${{action.expected_value.toFixed(3)}}</span>
                    <div style="color: #888; margin-top: 3px;">
                        (${{action.target.x.toFixed(0)}}, ${{action.target.y.toFixed(0)}}) - ${{(action.success_prob * 100).toFixed(0)}}%
                    </div>
                </div>
            `).join('');
        }}

        function highlightAction(index) {{
            const frame = frames[currentFrame];
            if (frame.available_actions && frame.available_actions[index]) {{
                highlightedAction = frame.available_actions[index];
                render();
            }}
        }}

        function clearHighlight() {{
            highlightedAction = null;
            render();
        }}

        // Simulation controls
        function stepForward() {{
            if (currentFrame < frames.length - 1) {{
                currentFrame++;
                render();
            }} else {{
                stopPlay();
            }}
        }}

        function stepBack() {{
            if (currentFrame > 0) {{
                currentFrame--;
                render();
            }}
        }}

        function reset() {{
            stopPlay();
            currentFrame = 0;
            render();
        }}

        function togglePlay() {{
            if (playing) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
        }}

        function startPlay() {{
            playing = true;
            document.getElementById('playBtn').textContent = 'Pause';
            playInterval = setInterval(() => {{
                if (currentFrame < frames.length - 1) {{
                    stepForward();
                }} else {{
                    stopPlay();
                }}
            }}, speed);
        }}

        function stopPlay() {{
            playing = false;
            document.getElementById('playBtn').textContent = 'Play';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        function updateSpeed() {{
            speed = 2200 - document.getElementById('speed').value;
            if (playing) {{
                stopPlay();
                startPlay();
            }}
        }}

        // Tab switching
        function switchTab(tab) {{
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
            event.target.classList.add('active');
            document.getElementById(tab + '-tab').style.display = 'block';
            render();
        }}

        // Editor functions
        function setTool(tool) {{
            currentTool = tool;
            document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tool-' + tool).classList.add('active');
        }}

        function clearPitch() {{
            editorState = {{
                attackers: [],
                defenders: [],
                ball: null,
                carrier: null
            }};
            selectedPlayer = null;
            document.getElementById('selectedPlayer').style.display = 'none';
            render();
        }}

        function loadScenario() {{
            // Load default 4-3-3 vs 4-4-2 scenario
            editorState = {{
                attackers: [
                    {{id: 'GK', x: -40, y: 0}},
                    {{id: 'CB1', x: -30, y: -10}},
                    {{id: 'CB2', x: -30, y: 10}},
                    {{id: 'LB', x: -25, y: -25}},
                    {{id: 'RB', x: -25, y: 25}},
                    {{id: 'CM1', x: -10, y: -8}},
                    {{id: 'CM2', x: -10, y: 8}},
                    {{id: 'CAM', x: 0, y: 0}},
                    {{id: 'LW', x: 10, y: -20}},
                    {{id: 'RW', x: 10, y: 20}},
                    {{id: 'ST', x: 15, y: 0}}
                ],
                defenders: [
                    {{id: 'D_GK', x: 45, y: 0}},
                    {{id: 'D_CB1', x: 35, y: -8}},
                    {{id: 'D_CB2', x: 35, y: 8}},
                    {{id: 'D_LB', x: 30, y: -25}},
                    {{id: 'D_RB', x: 30, y: 25}},
                    {{id: 'D_CM1', x: 20, y: -12}},
                    {{id: 'D_CM2', x: 20, y: 0}},
                    {{id: 'D_CM3', x: 20, y: 12}},
                    {{id: 'D_LM', x: 25, y: -20}},
                    {{id: 'D_RM', x: 25, y: 20}},
                    {{id: 'D_ST', x: 5, y: 0}}
                ],
                ball: {{x: -30, y: -10}},
                carrier: {{type: 'attacker', index: 1}}
            }};
            render();
        }}

        function deleteSelected() {{
            if (!selectedPlayer) return;

            if (selectedPlayer.type === 'attacker') {{
                editorState.attackers.splice(selectedPlayer.index, 1);
            }} else {{
                editorState.defenders.splice(selectedPlayer.index, 1);
            }}

            // Clear carrier if deleted
            if (editorState.carrier &&
                editorState.carrier.type === selectedPlayer.type &&
                editorState.carrier.index === selectedPlayer.index) {{
                editorState.carrier = null;
            }}

            selectedPlayer = null;
            document.getElementById('selectedPlayer').style.display = 'none';
            render();
        }}

        function runSimulation() {{
            alert('Simulation would run here with Python backend.\\n\\nTo run:\\n1. Export scenario as JSON\\n2. Run Python DecisionEngine\\n3. Load results back');
        }}

        // Canvas interaction
        canvas.addEventListener('mousedown', (e) => {{
            if (currentTab !== 'editor') return;

            const rect = canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const pos = canvasToPitch(cx, cy);

            if (currentTool === 'select') {{
                // Find clicked player
                let found = null;

                editorState.attackers.forEach((a, i) => {{
                    const dist = Math.sqrt((a.x - pos.x)**2 + (a.y - pos.y)**2);
                    if (dist < 3) {{
                        found = {{type: 'attacker', index: i}};
                    }}
                }});

                editorState.defenders.forEach((d, i) => {{
                    const dist = Math.sqrt((d.x - pos.x)**2 + (d.y - pos.y)**2);
                    if (dist < 3) {{
                        found = {{type: 'defender', index: i}};
                    }}
                }});

                selectedPlayer = found;
                dragging = !!found;

                if (found) {{
                    document.getElementById('selectedPlayer').style.display = 'block';
                    const player = found.type === 'attacker' ?
                        editorState.attackers[found.index] :
                        editorState.defenders[found.index];
                    document.getElementById('selectedId').textContent = player.id || (found.type[0].toUpperCase() + (found.index + 1));
                    document.getElementById('selectedPos').textContent = `(${{player.x.toFixed(1)}}, ${{player.y.toFixed(1)}})`;
                    document.getElementById('selectedTeam').textContent = found.type === 'attacker' ? 'Attack' : 'Defense';
                }} else {{
                    document.getElementById('selectedPlayer').style.display = 'none';
                }}
            }} else if (currentTool === 'attacker') {{
                editorState.attackers.push({{
                    id: 'A' + (editorState.attackers.length + 1),
                    x: pos.x,
                    y: pos.y
                }});
            }} else if (currentTool === 'defender') {{
                editorState.defenders.push({{
                    id: 'D' + (editorState.defenders.length + 1),
                    x: pos.x,
                    y: pos.y
                }});
            }} else if (currentTool === 'ball') {{
                editorState.ball = {{x: pos.x, y: pos.y}};
            }}

            render();
        }});

        canvas.addEventListener('mousemove', (e) => {{
            if (!dragging || !selectedPlayer) return;

            const rect = canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const pos = canvasToPitch(cx, cy);

            // Clamp to pitch
            pos.x = Math.max(-HALF_LENGTH, Math.min(HALF_LENGTH, pos.x));
            pos.y = Math.max(-HALF_WIDTH, Math.min(HALF_WIDTH, pos.y));

            if (selectedPlayer.type === 'attacker') {{
                editorState.attackers[selectedPlayer.index].x = pos.x;
                editorState.attackers[selectedPlayer.index].y = pos.y;
            }} else {{
                editorState.defenders[selectedPlayer.index].x = pos.x;
                editorState.defenders[selectedPlayer.index].y = pos.y;
            }}

            document.getElementById('selectedPos').textContent = `(${{pos.x.toFixed(1)}}, ${{pos.y.toFixed(1)}})`;
            render();
        }});

        canvas.addEventListener('mouseup', () => {{
            dragging = false;
        }});

        canvas.addEventListener('contextmenu', (e) => {{
            e.preventDefault();
            if (currentTab !== 'editor' || !selectedPlayer) return;

            // Set as ball carrier
            editorState.carrier = {{...selectedPlayer}};
            if (selectedPlayer.type === 'attacker') {{
                editorState.ball = {{
                    x: editorState.attackers[selectedPlayer.index].x,
                    y: editorState.attackers[selectedPlayer.index].y
                }};
            }} else {{
                editorState.ball = {{
                    x: editorState.defenders[selectedPlayer.index].x,
                    y: editorState.defenders[selectedPlayer.index].y
                }};
            }}
            render();
        }});

        // Calculate stats
        function updateStats() {{
            if (frames.length === 0) return;

            let success = 0, fail = 0, goals = 0;
            frames.forEach(f => {{
                if (f.result === 'success') success++;
                else if (f.result === 'goal') {{ goals++; success++; }}
                else if (f.result === 'intercepted' || f.result === 'tackled' || f.result === 'missed') fail++;
            }});

            document.getElementById('statSteps').textContent = frames.length;
            document.getElementById('statSuccess').textContent = success;
            document.getElementById('statFail').textContent = fail;
            document.getElementById('statGoals').textContent = goals;
        }}

        // Initialize
        if (frames.length > 0) {{
            document.getElementById('totalSteps').textContent = frames.length;
        }}
        updateStats();
        render();
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def run_and_create_ui():
    """Run simulation and generate tactical UI."""
    print("Creating scenario...")
    state = create_test_scenario()

    print("Running simulation...")
    engine = DecisionEngine()
    steps = engine.run(state, max_steps=15)

    print(f"Simulation completed: {len(steps)} steps")

    output_path = "tactical_ui.html"
    generate_tactical_ui(steps, state, output_path)
    print(f"Tactical UI saved to: {output_path}")
    print("Open this file in a web browser to view and interact.")

    return output_path


if __name__ == "__main__":
    run_and_create_ui()
