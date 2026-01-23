"""
Player and Ball Tracking App - Simplified

Click to select/track players across frames.
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
import base64

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.player_detector import PlayerDetector
from src.tracking.tracker import PlayerTracker

app = Flask(__name__)
CORS(app)

# Directories
VIDEO_DIR = PROJECT_ROOT / "labeling" / "video_cache"
EXPORT_DIR = Path(__file__).parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

# Global state
_detector = None
_tracker = None
_current_video = None
_video_cap = None
_video_cap_lock = threading.Lock()
_video_info = {}
_frame_cache = {}
_tracking_data = {}  # frame_num -> {players: [...], ball: {...}}
_selected_player_id = None
_next_manual_id = 1000  # Manual player IDs start at 1000
_manual_players = {}  # id -> last known position {frame, x, y, bbox}

def get_detector():
    """Lazy load player detector - using small model for better accuracy."""
    global _detector
    if _detector is None:
        print("Loading YOLOv8 small detector (more accurate)...")
        _detector = PlayerDetector(
            model_path='yolov8s.pt',  # Small model - better than nano
            confidence=0.10,  # Very low threshold to catch more players
            device='cpu'
        )
        print("Detector ready.")
    return _detector

def get_tracker():
    """Get or create tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PlayerTracker(track_thresh=0.4, frame_rate=30)
    return _tracker

def reset_state():
    """Reset all tracking state."""
    global _tracker, _tracking_data, _selected_player_id, _frame_cache
    _tracker = None
    _tracking_data = {}
    _selected_player_id = None
    _frame_cache = {}

def load_video(video_name):
    """Load a video file."""
    global _current_video, _video_cap, _video_info

    video_path = VIDEO_DIR / video_name
    if not video_path.exists():
        return None

    with _video_cap_lock:
        if _video_cap:
            _video_cap.release()
        _video_cap = cv2.VideoCapture(str(video_path))
        _current_video = video_name

        _video_info = {
            'name': video_name,
            'frame_count': int(_video_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': _video_cap.get(cv2.CAP_PROP_FPS),
            'width': int(_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        _video_info['duration'] = _video_info['frame_count'] / max(_video_info['fps'], 1)

    reset_state()
    return _video_info

def get_frame(frame_num):
    """Get a frame from current video."""
    if frame_num in _frame_cache:
        return _frame_cache[frame_num]

    with _video_cap_lock:
        _video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = _video_cap.read()

    if ret:
        # Keep cache small
        if len(_frame_cache) > 50:
            oldest = min(_frame_cache.keys())
            del _frame_cache[oldest]
        _frame_cache[frame_num] = frame
        return frame
    return None

def track_manual_player(player_id, from_frame, to_frame):
    """Track a manual player using optical flow, then snap to nearest YOLO detection."""
    if player_id not in _manual_players:
        return None

    info = _manual_players[player_id]
    if info['frame'] != from_frame:
        return None

    frame1 = get_frame(from_frame)
    frame2 = get_frame(to_frame)
    if frame1 is None or frame2 is None:
        return None

    # Get the center point to track
    x, y = info['x'], info['y']
    pt = np.array([[x, y]], dtype=np.float32)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Track with optical flow
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, pt.reshape(-1, 1, 2), None,
        winSize=(51, 51), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)
    )

    if status[0][0] == 1:
        new_x, new_y = float(new_pts[0][0][0]), float(new_pts[0][0][1])

        # Try to snap to a YOLO detection for better bbox
        detector = get_detector()
        detections = detector.detect(frame2)

        best_det = None
        best_dist = 60  # Max distance to snap

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) / 2, y2  # Use foot position
            dist = ((cx - new_x)**2 + (cy - new_y)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_det = det

        if best_det is not None:
            # Use YOLO bbox for accuracy
            x1, y1, x2, y2 = [float(c) for c in best_det.bbox]
            new_x = (x1 + x2) / 2
            new_y = y2
            bbox = [x1, y1, x2, y2]
        else:
            # Fallback to fixed size
            w, h = 35, 70
            bbox = [new_x - w/2, new_y - h, new_x + w/2, new_y]

        _manual_players[player_id] = {
            'frame': to_frame,
            'x': new_x,
            'y': new_y,
            'bbox': bbox
        }
        return _manual_players[player_id]
    return None

def filter_field_players(detections, frame_height, frame_width):
    """Filter to only players on the field, not coaches/crowd."""
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        h = y2 - y1
        w = x2 - x1
        cy = (y1 + y2) / 2  # center y

        # Skip if in bottom 10% of frame (likely coaches right at camera)
        if cy > frame_height * 0.90:
            continue

        # Skip if too tall (more than 40% of frame - very close to camera)
        if h > frame_height * 0.4:
            continue

        # Skip if too wide (not a standing person)
        if w > h * 1.5:
            continue

        filtered.append(det)
    return filtered

def detect_frame(frame_num, prev_frame=None):
    """Detect players in a frame using YOLO + simple ID assignment."""
    if frame_num in _tracking_data:
        return _tracking_data[frame_num]

    frame = get_frame(frame_num)
    if frame is None:
        return {'players': [], 'ball': None}

    detector = get_detector()

    # Detect all people
    detections = detector.detect(frame)

    print(f"[DETECT] Frame {frame_num}: {len(detections)} detections", flush=True)

    # Get previous frame players for ID matching
    prev_players = []
    if prev_frame is not None and prev_frame in _tracking_data:
        prev_players = _tracking_data[prev_frame]['players']

    # Convert detections to players with ID assignment
    players = []
    used_ids = set()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        h_det = y2 - y1
        w_det = x2 - x1

        # Skip if way too wide (not a person)
        if w_det > h_det * 2:
            continue

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        player_id = None
        best_dist = 80  # Max distance to consider a match

        # Try to match to a player from previous frame by position
        for prev_player in prev_players:
            if prev_player['id'] in used_ids:
                continue
            px, py = prev_player['center']
            dist = ((px - cx)**2 + (py - cy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                player_id = prev_player['id']

        if player_id is None:
            player_id = i + 1
            # Make sure ID is unique
            while player_id in used_ids:
                player_id += 1

        used_ids.add(player_id)

        players.append({
            'id': player_id,
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(det.confidence),
            'center': [float(cx), float(cy)],
            'foot': [float(cx), float(y2)]
        })

    # Track manual players - find matching YOLO detection for each
    print(f"[DETECT] Manual players to track: {list(_manual_players.keys())}, prev_frame={prev_frame}", flush=True)
    if _manual_players and prev_frame is not None:
        prev_frame_img = get_frame(prev_frame)
        print(f"[DETECT] Got prev_frame_img: {prev_frame_img is not None}", flush=True)

        for pid, info in list(_manual_players.items()):
            print(f"[DETECT] Checking manual #{pid}, last seen frame {info['frame']}", flush=True)
            # Only track if manual player was on the previous frame or recently
            if info['frame'] != prev_frame:
                if abs(info['frame'] - prev_frame) > 30:
                    print(f"[DETECT] Skipping #{pid} - too far from prev_frame", flush=True)
                    continue

            last_x, last_y = info['x'], info['y']
            search_x, search_y = last_x, last_y
            print(f"[DETECT] Manual #{pid} searching from ({last_x:.0f}, {last_y:.0f})", flush=True)

            # Use optical flow to predict where player moved
            if prev_frame_img is not None and frame is not None:
                pt = np.array([[last_x, last_y]], dtype=np.float32)
                gray1 = cv2.cvtColor(prev_frame_img, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    gray1, gray2, pt.reshape(-1, 1, 2), None,
                    winSize=(51, 51), maxLevel=4
                )
                if status[0][0] == 1:
                    search_x = float(new_pts[0][0][0])
                    search_y = float(new_pts[0][0][1])

            # Find YOLO detection closest to predicted position
            best_det = None
            best_dist = 150  # Larger search radius

            for det in detections:
                dx1, dy1, dx2, dy2 = det.bbox
                cx, cy = (dx1 + dx2) / 2, dy2  # foot position
                dist = ((cx - search_x)**2 + (cy - search_y)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            if best_det is not None:
                x1, y1, x2, y2 = [float(c) for c in best_det.bbox]
                cx, cy = (x1 + x2) / 2, y2

                # Check if this detection is already assigned to another player
                already_used = False
                for p in players:
                    px, py = p['center']
                    if ((px - (x1+x2)/2)**2 + (py - (y1+y2)/2)**2) ** 0.5 < 20:
                        already_used = True
                        break

                if not already_used:
                    # Update manual player position
                    _manual_players[pid] = {
                        'frame': frame_num,
                        'x': cx,
                        'y': cy,
                        'bbox': [x1, y1, x2, y2]
                    }

                    players.append({
                        'id': pid,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(best_det.confidence),
                        'center': [float((x1+x2)/2), float((y1+y2)/2)],
                        'foot': [float(cx), float(cy)],
                        'manual': True
                    })
                    print(f"[DETECT] Manual #{pid} tracked (dist: {best_dist:.0f}px)", flush=True)
                else:
                    # Detection exists but already used by another player - still track to that position
                    _manual_players[pid] = {
                        'frame': frame_num,
                        'x': cx,
                        'y': cy,
                        'bbox': [x1, y1, x2, y2]
                    }
                    print(f"[DETECT] Manual #{pid} matched detection already tracked by another player", flush=True)
            else:
                print(f"[DETECT] Manual #{pid} lost - no detection within 150px of ({search_x:.0f}, {search_y:.0f})", flush=True)

    print(f"[DETECT] Returning {len(players)} players", flush=True)
    _tracking_data[frame_num] = {'players': players, 'ball': None}
    return _tracking_data[frame_num]

def find_nearest_player(frame_num, x, y):
    """Find the player nearest to click point."""
    data = detect_frame(frame_num)
    players = data.get('players', [])

    print(f"[CLICK] Looking for player near ({x:.0f}, {y:.0f}), {len(players)} players in frame", flush=True)

    if not players:
        return None

    # Find player whose bbox contains the click, or nearest center
    best_id = None
    best_dist = float('inf')

    for p in players:
        x1, y1, x2, y2 = p['bbox']
        cx, cy = p['center']

        # Check if click is inside bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            print(f"[CLICK] Click inside player #{p['id']} bbox", flush=True)
            return p['id']

        # Otherwise find nearest
        dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = p['id']

    print(f"[CLICK] Nearest player #{best_id} at {best_dist:.0f}px", flush=True)

    # Return nearest if within 150px
    if best_dist < 150:
        return best_id
    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/videos')
def api_videos():
    """List available videos."""
    videos = []
    for f in VIDEO_DIR.glob('*.mp4'):
        videos.append({
            'name': f.name,
            'size_mb': round(f.stat().st_size / (1024*1024), 1)
        })
    return jsonify({'videos': sorted(videos, key=lambda x: x['name'])})

@app.route('/api/video/<path:video_name>/load', methods=['POST'])
def api_load_video(video_name):
    """Load a video."""
    info = load_video(video_name)
    if info:
        return jsonify(info)
    return jsonify({'error': 'Video not found'}), 404

@app.route('/api/frame/<int:frame_num>')
def api_get_frame(frame_num):
    """Get frame with tracking overlay."""
    try:
        frame = get_frame(frame_num)
        if frame is None:
            return jsonify({'error': 'Frame not found'}), 404

        # Get previous frame from query param for optical flow
        prev_frame = request.args.get('prev', None, type=int)

        # Detect/track (pass prev_frame for manual player tracking)
        data = detect_frame(frame_num, prev_frame)

        # Draw overlay
        frame_draw = frame.copy()

        for player in data.get('players', []):
            x1, y1, x2, y2 = [int(c) for c in player['bbox']]
            pid = player['id']

            # Yellow for selected, green for others
            if pid == _selected_player_id:
                color = (0, 255, 255)
                thickness = 3
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, thickness)
            label = f"#{pid}"
            cv2.putText(frame_draw, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw trajectory of selected player
        if _selected_player_id is not None:
            points = []
            for fn in range(max(0, frame_num-30), frame_num+1):
                if fn in _tracking_data:
                    for p in _tracking_data[fn]['players']:
                        if p['id'] == _selected_player_id:
                            points.append((int(p['foot'][0]), int(p['foot'][1])))
                            break
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame_draw, points[i-1], points[i], (0, 255, 255), 2)

        # Encode
        _, buffer = cv2.imencode('.jpg', frame_draw, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_b64 = base64.b64encode(buffer).decode()

        return jsonify({
            'frame': frame_num,
            'image': f'data:image/jpeg;base64,{img_b64}',
            'players': data.get('players', []),
            'selected': _selected_player_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'frame': frame_num}), 500

@app.route('/api/click', methods=['POST'])
def api_click():
    """Handle click - select nearest player."""
    global _selected_player_id

    data = request.json
    frame_num = data.get('frame')
    x = data.get('x')
    y = data.get('y')

    player_id = find_nearest_player(frame_num, x, y)

    if player_id is not None:
        _selected_player_id = player_id

    return jsonify({'selected': _selected_player_id})

@app.route('/api/select/<int:player_id>', methods=['POST'])
def api_select(player_id):
    """Select a player by ID."""
    global _selected_player_id
    _selected_player_id = player_id
    return jsonify({'selected': player_id})

@app.route('/api/deselect', methods=['POST'])
def api_deselect():
    """Deselect player."""
    global _selected_player_id
    _selected_player_id = None
    return jsonify({'selected': None})

@app.route('/api/add', methods=['POST'])
def api_add_player():
    """Click to select nearest player OR create a new manual player."""
    global _selected_player_id, _next_manual_id

    data = request.json
    frame_num = data.get('frame')
    x = data.get('x')
    y = data.get('y')

    print(f"[ADD] Click at frame {frame_num}, position ({x}, {y})", flush=True)

    # Make sure we have tracking data for this frame
    frame_data = detect_frame(frame_num)
    players = frame_data.get('players', [])
    frame = get_frame(frame_num)

    # Only select existing player if click is INSIDE their bbox
    for p in players:
        x1, y1, x2, y2 = p['bbox']
        if x1 <= x <= x2 and y1 <= y <= y2:
            _selected_player_id = p['id']
            print(f"[ADD] Selected existing player #{p['id']} (click inside bbox)", flush=True)
            return jsonify({'selected': _selected_player_id, 'player': p})

    # Try to find a YOLO detection near the click to get accurate bbox
    # Exclude detections that are already tracked
    detector = get_detector()
    detections = detector.detect(frame)

    best_det = None
    best_dist = 150  # Max distance to snap to YOLO detection

    for det in detections:
        dx1, dy1, dx2, dy2 = det.bbox
        cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2

        # Skip if this detection matches an already-tracked player
        already_tracked = False
        for p in players:
            px, py = p['center']
            if ((px - cx)**2 + (py - cy)**2) ** 0.5 < 40:
                already_tracked = True
                break

        if already_tracked:
            continue

        # Distance from click to detection center
        dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
        # Also check if click is inside detection bbox
        if dx1 <= x <= dx2 and dy1 <= y <= dy2:
            dist = 0
        if dist < best_dist:
            best_dist = dist
            best_det = det

    print(f"[ADD] Found {len(detections)} YOLO detections, best untracked: {best_dist:.0f}px away", flush=True)

    # Create new manual player
    player_id = _next_manual_id
    _next_manual_id += 1

    if best_det is not None:
        # Use YOLO detection bbox for accuracy
        bx1, by1, bx2, by2 = [float(c) for c in best_det.bbox]
        foot_x = (bx1 + bx2) / 2
        foot_y = by2
        bbox = [bx1, by1, bx2, by2]
        print(f"[ADD] Snapped to YOLO detection (dist: {best_dist:.0f}px)", flush=True)
    else:
        # Estimate size based on vertical position (higher = further = smaller)
        frame_h = frame.shape[0] if frame is not None else 1080
        scale = 0.3 + 0.7 * (y / frame_h)  # 0.3 at top, 1.0 at bottom
        w, h = int(20 * scale), int(50 * scale)
        foot_x, foot_y = x, y
        bbox = [x - w/2, y - h, x + w/2, y]
        print(f"[ADD] Using estimated size (scale: {scale:.2f})", flush=True)

    new_player = {
        'id': player_id,
        'bbox': bbox,
        'confidence': 1.0,
        'center': [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2],
        'foot': [foot_x, foot_y],
        'manual': True
    }

    # Add to tracking data
    _tracking_data[frame_num]['players'].append(new_player)

    # Store in manual players for tracking across frames
    _manual_players[player_id] = {
        'frame': frame_num,
        'x': foot_x,
        'y': foot_y,
        'bbox': bbox
    }

    print(f"[ADD] Created manual player #{player_id} at ({foot_x:.0f}, {foot_y:.0f})", flush=True)
    return jsonify({'created': player_id, 'player': new_player})

@app.route('/api/delete', methods=['POST'])
def api_delete_player():
    """Delete a player from tracking data."""
    global _selected_player_id

    data = request.json
    player_id = data.get('player_id', _selected_player_id)

    if player_id is None:
        return jsonify({'error': 'No player selected'}), 400

    deleted_from = []

    # Remove from all frames
    for frame_num, frame_data in _tracking_data.items():
        players = frame_data.get('players', [])
        original_count = len(players)
        frame_data['players'] = [p for p in players if p['id'] != player_id]
        if len(frame_data['players']) < original_count:
            deleted_from.append(frame_num)

    # Remove from manual players if it's a manual player
    if player_id in _manual_players:
        del _manual_players[player_id]

    # Deselect if we deleted the selected player
    if _selected_player_id == player_id:
        _selected_player_id = None

    print(f"[DELETE] Removed player #{player_id} from {len(deleted_from)} frames", flush=True)
    return jsonify({'deleted': player_id, 'frames': deleted_from})

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Clear tracking cache to re-detect with new settings."""
    global _tracking_data, _tracker, _detector, _manual_players
    _tracking_data = {}
    _tracker = None  # Reset tracker to get fresh IDs
    _manual_players = {}  # Clear manual players too
    return jsonify({'status': 'refreshed'})

@app.route('/api/export/csv', methods=['POST'])
def api_export_csv():
    """Export tracking data to CSV."""
    import csv

    if not _tracking_data:
        return jsonify({'error': 'No tracking data'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tracking_{timestamp}.csv"
    filepath = EXPORT_DIR / filename

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'player_id', 'x', 'y', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

        for frame_num in sorted(_tracking_data.keys()):
            for p in _tracking_data[frame_num].get('players', []):
                x1, y1, x2, y2 = p['bbox']
                fx, fy = p['foot']
                writer.writerow([frame_num, p['id'], fx, fy, x1, y1, x2, y2])

    return jsonify({'filename': filename, 'path': str(filepath)})

if __name__ == '__main__':
    print("=" * 50)
    print("Player Tracking App")
    print("=" * 50)
    print(f"Videos: {VIDEO_DIR}")
    print("Starting at http://localhost:5001")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
