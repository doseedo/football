"""Combined 3D ball tracker integrating detection, LSTM, and physics."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple
import numpy as np
import torch
from loguru import logger

from ..detection.ball_detector import BallDetector, BallDetection
from .trajectory_lstm import TrajectoryLSTM, CanonicalRepresentation, create_context_features
from .physics_model import BallPhysicsModel, Ball3DPosition, PhysicsConstraints


@dataclass
class Ball3DState:
    """Complete state of ball in 3D."""
    position_2d: Tuple[float, float]  # Pixel coordinates
    position_3d: Ball3DPosition  # 3D world coordinates
    confidence_2d: float  # 2D detection confidence
    confidence_3d: float  # 3D estimation confidence
    is_interpolated: bool  # Whether 2D was interpolated
    frame_idx: int  # Frame number
    timestamp: float  # Time in seconds


class Ball3DTracker:
    """
    Combined 3D ball tracker.

    Integrates:
    - 2D ball detection (BallDetector)
    - 3D trajectory estimation (TrajectoryLSTM)
    - Physics constraints (BallPhysicsModel)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        detection_confidence: float = 0.2,
        temporal_window: int = 10,
        sequence_length: int = 20,
        device: str = 'cuda',
        fps: float = 25.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ):
        """
        Initialize 3D ball tracker.

        Args:
            model_path: Path to trained LSTM model weights
            detection_confidence: Confidence threshold for 2D detection
            temporal_window: Window for temporal consistency
            sequence_length: Sequence length for LSTM
            device: Device for neural network
            fps: Frame rate
            pitch_length: Pitch length in meters
            pitch_width: Pitch width in meters
        """
        self.device = device
        self.fps = fps
        self.dt = 1.0 / fps
        self.sequence_length = sequence_length
        self.temporal_window = temporal_window
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Initialize 2D ball detector
        logger.info("Initializing 2D ball detector...")
        self.ball_detector_2d = BallDetector(
            confidence=detection_confidence,
            temporal_window=temporal_window,
            device=device
        )

        # Initialize LSTM model
        logger.info("Initializing 3D trajectory LSTM...")
        self.lstm_model = TrajectoryLSTM(
            input_dim=2,
            context_dim=11,  # 9 for homography + 2 for camera params
            hidden_dim=256,
            num_layers=2,
            output_dim=3,
            predict_uncertainty=True
        ).to(device)

        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            logger.info(f"Loading LSTM weights from {model_path}")
            self.lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            logger.warning("No pretrained LSTM model found. Using random initialization.")

        self.lstm_model.eval()

        # Initialize physics model
        self.physics_model = BallPhysicsModel(fps=fps)

        # Initialize canonical representation
        self.canonical = CanonicalRepresentation(pitch_length, pitch_width)

        # State tracking
        self.position_buffer_2d: Deque[Tuple[float, float]] = deque(maxlen=sequence_length)
        self.position_buffer_3d: Deque[Ball3DPosition] = deque(maxlen=sequence_length)
        self.trajectory_history: List[Ball3DState] = []
        self.frame_count = 0

        # Homography and camera info
        self.homography: Optional[np.ndarray] = None
        self.camera_height: float = 15.0
        self.camera_angle: float = 30.0

    def set_calibration(
        self,
        homography: np.ndarray,
        camera_height: float = 15.0,
        camera_angle: float = 30.0
    ):
        """
        Set camera calibration parameters.

        Args:
            homography: (3, 3) homography matrix
            camera_height: Camera height above pitch in meters
            camera_angle: Camera tilt angle in degrees
        """
        self.homography = homography
        self.camera_height = camera_height
        self.camera_angle = camera_angle
        logger.info("Camera calibration set for 3D tracking")

    def track(
        self,
        frame: np.ndarray,
        ball_2d_position: Optional[Tuple[float, float]] = None,
        player_bboxes: Optional[List[np.ndarray]] = None
    ) -> Optional[Ball3DState]:
        """
        Track ball in current frame and estimate 3D position.

        Args:
            frame: Current video frame
            ball_2d_position: Pre-detected 2D position (if None, will detect)
            player_bboxes: Player bounding boxes for occlusion handling

        Returns:
            Ball3DState with complete 3D information, or None if not detected
        """
        self.frame_count += 1
        timestamp = self.frame_count / self.fps

        # Step 1: Get 2D detection
        if ball_2d_position is None:
            ball_detection = self.ball_detector_2d.detect(frame, player_bboxes)
            if ball_detection is None:
                # No detection, try to maintain trajectory with physics
                return self._predict_from_physics(timestamp)

            position_2d = ball_detection.position
            confidence_2d = ball_detection.confidence
            is_interpolated = ball_detection.is_interpolated
        else:
            position_2d = ball_2d_position
            confidence_2d = 1.0
            is_interpolated = False

        # Add to buffer
        self.position_buffer_2d.append(position_2d)

        # Step 2: Check if we have enough data for LSTM
        if len(self.position_buffer_2d) < 3:
            # Not enough history, use simple homography projection
            position_3d = self._simple_3d_estimation(position_2d, timestamp)
            confidence_3d = confidence_2d * 0.5  # Lower confidence without LSTM

        else:
            # Step 3: Use LSTM for 3D estimation
            position_3d, confidence_3d = self._lstm_3d_estimation(
                list(self.position_buffer_2d),
                timestamp
            )

        # Step 4: Apply physics constraints
        if len(self.position_buffer_3d) > 0:
            position_3d = self._apply_physics_constraints(position_3d)

        # Add to 3D buffer
        self.position_buffer_3d.append(position_3d)

        # Create state
        state = Ball3DState(
            position_2d=position_2d,
            position_3d=position_3d,
            confidence_2d=confidence_2d,
            confidence_3d=confidence_3d,
            is_interpolated=is_interpolated,
            frame_idx=self.frame_count,
            timestamp=timestamp
        )

        self.trajectory_history.append(state)

        return state

    def _simple_3d_estimation(
        self,
        position_2d: Tuple[float, float],
        timestamp: float
    ) -> Ball3DPosition:
        """
        Simple 3D estimation using homography (assumes ball on ground).

        Args:
            position_2d: 2D pixel position
            timestamp: Current timestamp

        Returns:
            Ball3DPosition (with z=0)
        """
        if self.homography is None:
            # No homography, return dummy position
            return Ball3DPosition(
                x=0.0, y=0.0, z=0.0,
                timestamp=timestamp,
                confidence=0.0
            )

        # Apply homography
        point = np.array([[position_2d[0], position_2d[1]]], dtype=np.float32)
        point_h = np.concatenate([point, np.ones((1, 1))], axis=1)
        transformed = (self.homography @ point_h.T).T
        world_point = transformed[0, :2] / transformed[0, 2]

        return Ball3DPosition(
            x=float(world_point[0]),
            y=float(world_point[1]),
            z=0.11,  # Ball radius (on ground)
            timestamp=timestamp,
            confidence=0.7
        )

    def _lstm_3d_estimation(
        self,
        positions_2d: List[Tuple[float, float]],
        timestamp: float
    ) -> Tuple[Ball3DPosition, float]:
        """
        Estimate 3D position using LSTM model.

        Args:
            positions_2d: Recent 2D positions
            timestamp: Current timestamp

        Returns:
            position_3d: Estimated 3D position
            confidence: Estimation confidence
        """
        if self.homography is None:
            return self._simple_3d_estimation(positions_2d[-1], timestamp), 0.5

        # Prepare input
        positions_array = np.array(positions_2d, dtype=np.float32)

        # Pad or truncate to sequence length
        if len(positions_array) < self.sequence_length:
            # Pad with first position
            padding = np.repeat(positions_array[0:1], self.sequence_length - len(positions_array), axis=0)
            positions_array = np.concatenate([padding, positions_array], axis=0)
        else:
            positions_array = positions_array[-self.sequence_length:]

        # Create context features
        context = create_context_features(
            self.homography,
            self.camera_height,
            self.camera_angle
        )

        # Convert to tensors
        positions_tensor = torch.from_numpy(positions_array).unsqueeze(0).to(self.device)
        context_tensor = torch.from_numpy(context).unsqueeze(0).to(self.device)

        # LSTM inference
        with torch.no_grad():
            output = self.lstm_model(positions_tensor, context_tensor)
            pred_3d = output['position_3d'][0, -1].cpu().numpy()  # Last position
            if output.get('uncertainty') is not None:
                uncertainty = output['uncertainty'][0, -1].cpu().numpy()
                # Convert uncertainty to confidence (inverse relationship)
                confidence = float(1.0 / (1.0 + np.mean(uncertainty)))
            else:
                confidence = 0.8

        # Create Ball3DPosition
        position_3d = Ball3DPosition(
            x=float(pred_3d[0]),
            y=float(pred_3d[1]),
            z=float(max(0.0, pred_3d[2])),  # Ensure non-negative height
            timestamp=timestamp,
            confidence=confidence
        )

        return position_3d, confidence

    def _apply_physics_constraints(self, position: Ball3DPosition) -> Ball3DPosition:
        """
        Apply physics constraints to estimated position.

        Args:
            position: Unconstrained position

        Returns:
            Constrained position
        """
        # Get recent trajectory
        recent_positions = list(self.position_buffer_3d)[-5:] + [position]

        # Apply physics model
        constrained = self.physics_model.apply_physics_constraints(
            recent_positions,
            smooth=True
        )

        return constrained[-1]

    def _predict_from_physics(self, timestamp: float) -> Optional[Ball3DState]:
        """
        Predict ball position using physics when detection fails.

        Args:
            timestamp: Current timestamp

        Returns:
            Predicted Ball3DState or None
        """
        if len(self.position_buffer_3d) < 2:
            return None

        # Use last known position and velocity
        last_position = self.position_buffer_3d[-1]

        # Predict next position
        predicted = self.physics_model.predict_next_position(last_position, steps=1)

        # Project back to 2D (for visualization)
        if self.homography is not None:
            # Inverse homography
            H_inv = np.linalg.inv(self.homography)
            world_point = np.array([predicted.x, predicted.y, 1.0])
            pixel_point = H_inv @ world_point
            position_2d = (
                float(pixel_point[0] / pixel_point[2]),
                float(pixel_point[1] / pixel_point[2])
            )
        else:
            position_2d = (0.0, 0.0)

        state = Ball3DState(
            position_2d=position_2d,
            position_3d=predicted,
            confidence_2d=0.3,  # Low confidence for prediction
            confidence_3d=predicted.confidence,
            is_interpolated=True,
            frame_idx=self.frame_count,
            timestamp=timestamp
        )

        self.trajectory_history.append(state)

        return state

    def get_trajectory_history(
        self,
        num_frames: Optional[int] = None
    ) -> List[Ball3DState]:
        """
        Get trajectory history.

        Args:
            num_frames: Number of recent frames (all if None)

        Returns:
            List of Ball3DState
        """
        if num_frames is None:
            return self.trajectory_history
        else:
            return self.trajectory_history[-num_frames:]

    def get_current_height(self) -> float:
        """Get current ball height estimate."""
        if len(self.position_buffer_3d) > 0:
            return self.position_buffer_3d[-1].z
        return 0.0

    def is_ball_aerial(self, threshold: float = 0.5) -> bool:
        """
        Check if ball is currently aerial.

        Args:
            threshold: Height threshold in meters

        Returns:
            True if ball is above threshold
        """
        return self.get_current_height() > threshold

    def reset(self):
        """Reset tracker state."""
        self.ball_detector_2d.reset()
        self.position_buffer_2d.clear()
        self.position_buffer_3d.clear()
        self.trajectory_history.clear()
        self.frame_count = 0
        logger.info("3D ball tracker reset")

    def export_trajectory_3d(self, filepath: str):
        """
        Export 3D trajectory to file.

        Args:
            filepath: Output file path (CSV or JSON)
        """
        import pandas as pd
        import json

        data = []
        for state in self.trajectory_history:
            data.append({
                'frame': state.frame_idx,
                'timestamp': state.timestamp,
                'x_2d': state.position_2d[0],
                'y_2d': state.position_2d[1],
                'x_3d': state.position_3d.x,
                'y_3d': state.position_3d.y,
                'z_3d': state.position_3d.z,
                'confidence_2d': state.confidence_2d,
                'confidence_3d': state.confidence_3d,
                'is_interpolated': state.is_interpolated,
                'is_aerial': state.position_3d.z > 0.5
            })

        filepath = Path(filepath)
        if filepath.suffix == '.csv':
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        elif filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        logger.info(f"3D trajectory exported to {filepath}")

    def get_statistics(self) -> dict:
        """Get tracking statistics."""
        if len(self.trajectory_history) == 0:
            return {}

        heights = [s.position_3d.z for s in self.trajectory_history]
        confidences = [s.confidence_3d for s in self.trajectory_history]
        aerial_frames = sum(1 for s in self.trajectory_history if s.position_3d.z > 0.5)

        return {
            'total_frames': len(self.trajectory_history),
            'aerial_frames': aerial_frames,
            'aerial_percentage': aerial_frames / len(self.trajectory_history) * 100,
            'max_height': max(heights),
            'avg_height': np.mean(heights),
            'avg_confidence_3d': np.mean(confidences),
            'interpolated_frames': sum(1 for s in self.trajectory_history if s.is_interpolated)
        }
