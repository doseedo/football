"""
Web-based visualizer for the Decision Engine.

Generates an HTML file that animates the simulation.
"""

import json
from typing import List
from .action_executor import DecisionStep, DecisionEngine, create_test_scenario, ActionResult
from .state_scoring import GameState


def generate_visualization_html(steps: List[DecisionStep], output_path: str = "simulation.html"):
    """Generate an HTML file that visualizes the simulation."""

    # Convert steps to JSON-serializable format
    frames = []
    for i, step in enumerate(steps):
        frame = {
            "step": i + 1,
            "ball": {"x": step.state_before.ball_position.x, "y": step.state_before.ball_position.y},
            "carrier": step.state_before.ball_carrier.id if step.state_before.ball_carrier else None,
            "attackers": [
                {"id": p.id, "x": p.position.x, "y": p.position.y}
                for p in step.state_before.attackers
            ],
            "defenders": [
                {"id": p.id, "x": p.position.x, "y": p.position.y}
                for p in step.state_before.defenders
            ],
            "action": None,
            "result": None,
            "message": ""
        }

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
                {"id": p.id, "x": p.position.x, "y": p.position.y}
                for p in last.state_after.attackers
            ],
            "defenders": [
                {"id": p.id, "x": p.position.x, "y": p.position.y}
                for p in last.state_after.defenders
            ],
            "action": None,
            "result": "end",
            "message": "Simulation complete"
        }
        frames.append(final_frame)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Football Decision Engine</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{
            color: #4ecca3;
            margin-bottom: 10px;
        }}
        .container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .pitch-container {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
        }}
        #pitch {{
            background: #2d5a27;
            border: 3px solid #fff;
            border-radius: 5px;
        }}
        .info-panel {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            min-width: 300px;
        }}
        .controls {{
            margin: 20px 0;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            background: #4ecca3;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }}
        button:hover {{
            background: #3db892;
        }}
        button:disabled {{
            background: #666;
            cursor: not-allowed;
        }}
        .step-info {{
            margin: 10px 0;
            padding: 10px;
            background: #0f3460;
            border-radius: 5px;
        }}
        .action {{
            color: #4ecca3;
            font-weight: bold;
        }}
        .result-success {{
            color: #4ecca3;
        }}
        .result-fail {{
            color: #e94560;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 14px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        input[type="range"] {{
            width: 100px;
        }}
    </style>
</head>
<body>
    <h1>Football Decision Engine</h1>

    <div class="controls">
        <button id="playBtn" onclick="togglePlay()">▶ Play</button>
        <button onclick="stepBack()" id="backBtn">◀ Back</button>
        <button onclick="stepForward()" id="fwdBtn">Next ▶</button>
        <button onclick="reset()">↺ Reset</button>
        <div class="speed-control">
            <span>Speed:</span>
            <input type="range" id="speed" min="200" max="2000" value="1000" onchange="updateSpeed()">
        </div>
    </div>

    <div class="container">
        <div class="pitch-container">
            <canvas id="pitch" width="700" height="450"></canvas>
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
            </div>
        </div>

        <div class="info-panel">
            <h3>Step <span id="stepNum">1</span> / {len(frames)}</h3>
            <div class="step-info">
                <div><strong>Ball Carrier:</strong> <span id="carrier">-</span></div>
                <div><strong>Ball Position:</strong> <span id="ballPos">-</span></div>
            </div>
            <div class="step-info">
                <div><strong>Action:</strong> <span id="actionType" class="action">-</span></div>
                <div><strong>Target:</strong> <span id="actionTarget">-</span></div>
                <div><strong>Success Prob:</strong> <span id="successProb">-</span></div>
                <div><strong>Expected Value:</strong> <span id="expValue">-</span></div>
            </div>
            <div class="step-info">
                <div><strong>Result:</strong> <span id="result">-</span></div>
                <div id="message"></div>
            </div>
        </div>
    </div>

    <script>
        const frames = {json.dumps(frames)};
        let currentFrame = 0;
        let playing = false;
        let playInterval = null;
        let speed = 1000;

        const canvas = document.getElementById('pitch');
        const ctx = canvas.getContext('2d');

        // Pitch dimensions (in meters, centered at 0,0)
        const PITCH_LENGTH = 105;
        const PITCH_WIDTH = 68;
        const HALF_LENGTH = 52.5;
        const HALF_WIDTH = 34;

        // Canvas scaling
        const scaleX = canvas.width / PITCH_LENGTH;
        const scaleY = canvas.height / PITCH_WIDTH;

        function pitchToCanvas(x, y) {{
            return {{
                x: (x + HALF_LENGTH) * scaleX,
                y: (HALF_WIDTH - y) * scaleY  // Flip Y axis
            }};
        }}

        function drawPitch() {{
            // Green background
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

            // Penalty areas
            const penaltyLength = 16.5;
            const penaltyWidth = 40.3;

            // Left penalty area
            const leftPenalty = pitchToCanvas(-HALF_LENGTH, penaltyWidth/2);
            ctx.strokeRect(leftPenalty.x, leftPenalty.y, penaltyLength * scaleX, penaltyWidth * scaleY);

            // Right penalty area
            const rightPenalty = pitchToCanvas(HALF_LENGTH - penaltyLength, penaltyWidth/2);
            ctx.strokeRect(rightPenalty.x, rightPenalty.y, penaltyLength * scaleX, penaltyWidth * scaleY);

            // Goals
            ctx.fillStyle = '#fff';
            const goalWidth = 7.32;
            const leftGoal = pitchToCanvas(-HALF_LENGTH - 2, goalWidth/2);
            ctx.fillRect(leftGoal.x, leftGoal.y, 2 * scaleX, goalWidth * scaleY);

            const rightGoal = pitchToCanvas(HALF_LENGTH, goalWidth/2);
            ctx.fillRect(rightGoal.x, rightGoal.y, 2 * scaleX, goalWidth * scaleY);
        }}

        function drawPlayer(x, y, color, label, hasBall) {{
            const pos = pitchToCanvas(x, y);

            // Player circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = hasBall ? '#f1c40f' : '#fff';
            ctx.lineWidth = hasBall ? 3 : 1;
            ctx.stroke();

            // Label
            ctx.fillStyle = '#fff';
            ctx.font = '9px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(label, pos.x, pos.y + 3);
        }}

        function drawBall(x, y) {{
            const pos = pitchToCanvas(x, y);
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#f1c40f';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            ctx.stroke();
        }}

        function drawAction(frame) {{
            if (!frame.action) return;

            const from = pitchToCanvas(frame.ball.x, frame.ball.y);
            const to = pitchToCanvas(frame.action.target.x, frame.action.target.y);

            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = frame.result === 'success' || frame.result === 'goal'
                ? 'rgba(46, 204, 113, 0.8)'
                : 'rgba(231, 76, 60, 0.8)';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Arrow head
            const angle = Math.atan2(to.y - from.y, to.x - from.x);
            ctx.beginPath();
            ctx.moveTo(to.x, to.y);
            ctx.lineTo(to.x - 10 * Math.cos(angle - 0.5), to.y - 10 * Math.sin(angle - 0.5));
            ctx.lineTo(to.x - 10 * Math.cos(angle + 0.5), to.y - 10 * Math.sin(angle + 0.5));
            ctx.closePath();
            ctx.fill();
        }}

        function render() {{
            const frame = frames[currentFrame];

            drawPitch();

            // Draw action line first (behind players)
            drawAction(frame);

            // Draw defenders
            frame.defenders.forEach(d => {{
                drawPlayer(d.x, d.y, '#e74c3c', d.id.replace('D_', ''), false);
            }});

            // Draw attackers
            frame.attackers.forEach(a => {{
                const hasBall = a.id === frame.carrier;
                drawPlayer(a.x, a.y, '#3498db', a.id, hasBall);
            }});

            // Draw ball
            drawBall(frame.ball.x, frame.ball.y);

            // Update info panel
            document.getElementById('stepNum').textContent = frame.step;
            document.getElementById('carrier').textContent = frame.carrier || '-';
            document.getElementById('ballPos').textContent =
                `(${{frame.ball.x.toFixed(1)}}, ${{frame.ball.y.toFixed(1)}})`;

            if (frame.action) {{
                document.getElementById('actionType').textContent = frame.action.type;
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
            resultEl.className = frame.result === 'success' || frame.result === 'goal'
                ? 'result-success' : 'result-fail';
            document.getElementById('message').textContent = frame.message;

            // Update button states
            document.getElementById('backBtn').disabled = currentFrame === 0;
            document.getElementById('fwdBtn').disabled = currentFrame >= frames.length - 1;
        }}

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
            document.getElementById('playBtn').textContent = '⏸ Pause';
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
            document.getElementById('playBtn').textContent = '▶ Play';
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

        // Initial render
        render();
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def run_and_visualize():
    """Run simulation and generate visualization."""
    print("Creating scenario...")
    state = create_test_scenario()

    print("Running simulation...")
    engine = DecisionEngine()
    steps = engine.run(state, max_steps=15)

    print(f"Simulation completed: {len(steps)} steps")

    output_path = "simulation.html"
    generate_visualization_html(steps, output_path)
    print(f"Visualization saved to: {output_path}")
    print("Open this file in a web browser to view the simulation.")

    return output_path


if __name__ == "__main__":
    run_and_visualize()
