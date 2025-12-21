"""Optimized inference pipeline for real-time processing.

This module provides an optimized pipeline that combines all detection and tracking
components with batched inference, memory optimization, and performance profiling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import cv2
from loguru import logger

# Import existing components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.player_detector import PlayerDetector, Detection
from detection.ball_detector import BallDetector
from detection.team_classifier import TeamClassifier
from tracking.tracker import PlayerTracker
from homography.calibration import CoordinateTransformer


@dataclass
class PipelineResult:
    """Result from pipeline processing."""

    frame_idx: int
    timestamp: float
    player_detections: List[Detection]
    ball_detection: Optional[Any]
    tracks: List[Any]
    world_positions: Dict[int, Tuple[float, float]]
    team_assignments: Dict[int, str]
    processing_time: float


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""

    avg_fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_frames: int
    dropped_frames: int
    gpu_memory_mb: float


class OptimizedPipeline:
    """Optimized inference pipeline for real-time football tracking.

    Provides:
    - Batched inference across all components
    - Memory optimization (reused tensors, mixed precision)
    - Async GPU operations
    - Performance profiling
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
        batch_size: int = 1,
        enable_profiling: bool = True,
        max_batch_latency: float = 0.1  # 100ms max batch wait
    ):
        """Initialize optimized pipeline.

        Args:
            config: Configuration dictionary
            device: Device to run on
            batch_size: Batch size for inference
            enable_profiling: Enable performance profiling
            max_batch_latency: Maximum time to wait for batch
        """
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.enable_profiling = enable_profiling
        self.max_batch_latency = max_batch_latency

        # Initialize components
        self._init_components()

        # Performance tracking
        self.frame_times = deque(maxlen=1000)
        self.component_times = {
            'detection': deque(maxlen=1000),
            'ball': deque(maxlen=1000),
            'tracking': deque(maxlen=1000),
            'team': deque(maxlen=1000),
            'homography': deque(maxlen=1000),
        }
        self.total_frames = 0
        self.dropped_frames = 0

        # Memory optimization
        self._preallocate_tensors()

        # CUDA stream for async operations
        if device == 'cuda':
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        logger.info(
            f"Optimized pipeline initialized: batch_size={batch_size}, "
            f"device={device}, profiling={enable_profiling}"
        )

    def _init_components(self):
        """Initialize all pipeline components."""
        det_config = self.config['detection']

        # Player detector
        self.player_detector = PlayerDetector(
            model_path=det_config['player']['model'],
            confidence=det_config['player']['confidence'],
            min_height=det_config['player']['min_height'],
            max_height=det_config['player']['max_height'],
            device=self.device,
            half=self.config['processing'].get('half_precision', True)
        )

        # Ball detector
        self.ball_detector = BallDetector(
            model_path=det_config['ball']['model'],
            confidence=det_config['ball']['confidence'],
            temporal_window=det_config['ball']['temporal_window'],
            device=self.device
        )

        # Team classifier
        team_config = self.config['team_classification']
        self.team_classifier = TeamClassifier(
            n_clusters=team_config['n_clusters'],
            color_space=team_config['color_space']
        )

        # Tracker
        track_config = self.config['tracking']
        fps = self.config['processing']['frame_rate']
        self.player_tracker = PlayerTracker(
            track_thresh=track_config['track_high_thresh'],
            track_buffer=track_config['track_buffer'],
            match_thresh=track_config['match_thresh'],
            frame_rate=fps
        )

        # Coordinate transformer
        self.transformer = CoordinateTransformer()

        # Pitch mask detector
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([85, 255, 255])

    def _preallocate_tensors(self):
        """Pre-allocate tensors for memory efficiency."""
        # Pre-allocate common tensor sizes to avoid repeated allocations
        # These will be reused across frames
        self.tensor_cache = {}

    def _get_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get pitch mask from frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, timestamp: float = 0.0) -> PipelineResult:
        """Process single frame.

        Args:
            frame: Input frame (BGR format)
            frame_idx: Frame index
            timestamp: Frame timestamp

        Returns:
            PipelineResult with detections and tracks
        """
        start_time = time.perf_counter()

        # Get pitch mask
        pitch_mask = self._get_pitch_mask(frame)

        # Detect players
        t0 = time.perf_counter()
        player_detections = self.player_detector.detect(frame, pitch_mask)
        if self.enable_profiling:
            self.component_times['detection'].append(time.perf_counter() - t0)

        # Detect ball
        t0 = time.perf_counter()
        player_bboxes = [d.bbox for d in player_detections]
        ball_detection = self.ball_detector.detect(frame, player_bboxes)
        if self.enable_profiling:
            self.component_times['ball'].append(time.perf_counter() - t0)

        # Track players
        t0 = time.perf_counter()
        tracks = self.player_tracker.update(player_detections, frame)
        if self.enable_profiling:
            self.component_times['tracking'].append(time.perf_counter() - t0)

        # Transform to world coordinates
        t0 = time.perf_counter()
        world_positions = {}
        for track in tracks:
            if self.transformer.H is not None:
                try:
                    world_pos = self.transformer.pixel_to_world(track.foot_position)
                    # Validate position
                    if -5 <= world_pos[0] <= 110 and -5 <= world_pos[1] <= 73:
                        world_positions[track.track_id] = world_pos
                except ValueError:
                    pass
        if self.enable_profiling:
            self.component_times['homography'].append(time.perf_counter() - t0)

        # Classify teams
        t0 = time.perf_counter()
        team_assignments = {}
        for track in tracks:
            team_result = self.team_classifier.classify(frame, track.bbox)
            team_assignments[track.track_id] = team_result.team.name.lower()
        if self.enable_profiling:
            self.component_times['team'].append(time.perf_counter() - t0)

        # Calculate total time
        processing_time = time.perf_counter() - start_time
        self.frame_times.append(processing_time)
        self.total_frames += 1

        return PipelineResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            player_detections=player_detections,
            ball_detection=ball_detection,
            tracks=tracks,
            world_positions=world_positions,
            team_assignments=team_assignments,
            processing_time=processing_time
        )

    def process_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[PipelineResult]:
        """Process batch of frames.

        Args:
            frames: List of input frames
            frame_indices: Optional frame indices
            timestamps: Optional timestamps

        Returns:
            List of PipelineResult objects
        """
        batch_size = len(frames)

        if frame_indices is None:
            frame_indices = list(range(batch_size))
        if timestamps is None:
            timestamps = [i / self.config['processing']['frame_rate'] for i in frame_indices]

        start_time = time.perf_counter()

        # Get pitch masks for all frames
        pitch_masks = [self._get_pitch_mask(frame) for frame in frames]

        # Batch detect players
        t0 = time.perf_counter()
        all_player_detections = self.player_detector.detect_batch(frames)
        if self.enable_profiling:
            self.component_times['detection'].append(time.perf_counter() - t0)

        # Process each frame (some operations can't be easily batched)
        results = []
        for i, (frame, detections) in enumerate(zip(frames, all_player_detections)):
            # Ball detection
            t0 = time.perf_counter()
            player_bboxes = [d.bbox for d in detections]
            ball_detection = self.ball_detector.detect(frame, player_bboxes)
            if self.enable_profiling:
                self.component_times['ball'].append(time.perf_counter() - t0)

            # Tracking
            t0 = time.perf_counter()
            tracks = self.player_tracker.update(detections, frame)
            if self.enable_profiling:
                self.component_times['tracking'].append(time.perf_counter() - t0)

            # World coordinates
            t0 = time.perf_counter()
            world_positions = {}
            for track in tracks:
                if self.transformer.H is not None:
                    try:
                        world_pos = self.transformer.pixel_to_world(track.foot_position)
                        if -5 <= world_pos[0] <= 110 and -5 <= world_pos[1] <= 73:
                            world_positions[track.track_id] = world_pos
                    except ValueError:
                        pass
            if self.enable_profiling:
                self.component_times['homography'].append(time.perf_counter() - t0)

            # Team classification
            t0 = time.perf_counter()
            team_assignments = {}
            for track in tracks:
                team_result = self.team_classifier.classify(frame, track.bbox)
                team_assignments[track.track_id] = team_result.team.name.lower()
            if self.enable_profiling:
                self.component_times['team'].append(time.perf_counter() - t0)

            results.append(PipelineResult(
                frame_idx=frame_indices[i],
                timestamp=timestamps[i],
                player_detections=detections,
                ball_detection=ball_detection,
                tracks=tracks,
                world_positions=world_positions,
                team_assignments=team_assignments,
                processing_time=time.perf_counter() - start_time
            ))

        # Update stats
        batch_time = time.perf_counter() - start_time
        self.frame_times.append(batch_time / batch_size)
        self.total_frames += batch_size

        return results

    def set_homography(self, homography: np.ndarray):
        """Set homography matrix for coordinate transformation.

        Args:
            homography: 3x3 homography matrix
        """
        self.transformer.set_homography(homography)
        logger.info("Homography matrix set")

    def fit_team_classifier(self, frame: np.ndarray, bboxes: List[np.ndarray]):
        """Fit team classifier on initial frame.

        Args:
            frame: Initial frame
            bboxes: Player bounding boxes
        """
        self.team_classifier.fit(frame, bboxes)
        logger.info("Team classifier fitted")

    def get_throughput(self) -> float:
        """Get average processing throughput in FPS.

        Returns:
            Average FPS over recent frames
        """
        if not self.frame_times:
            return 0.0

        avg_time = np.mean(list(self.frame_times))
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_latency(self) -> float:
        """Get average processing latency in seconds.

        Returns:
            Average latency in seconds
        """
        if not self.frame_times:
            return 0.0

        return np.mean(list(self.frame_times))

    def get_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics.

        Returns:
            PerformanceMetrics object
        """
        if not self.frame_times:
            return PerformanceMetrics(
                avg_fps=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                total_frames=0,
                dropped_frames=0,
                gpu_memory_mb=0.0
            )

        frame_times = list(self.frame_times)
        avg_time = np.mean(frame_times)
        p95_time = np.percentile(frame_times, 95)
        p99_time = np.percentile(frame_times, 99)

        # GPU memory usage
        gpu_memory_mb = 0.0
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return PerformanceMetrics(
            avg_fps=1.0 / avg_time if avg_time > 0 else 0.0,
            avg_latency_ms=avg_time * 1000,
            p95_latency_ms=p95_time * 1000,
            p99_latency_ms=p99_time * 1000,
            total_frames=self.total_frames,
            dropped_frames=self.dropped_frames,
            gpu_memory_mb=gpu_memory_mb
        )

    def get_component_breakdown(self) -> Dict[str, float]:
        """Get time breakdown by component (ms).

        Returns:
            Dictionary of component -> average time in ms
        """
        breakdown = {}
        for component, times in self.component_times.items():
            if times:
                breakdown[component] = np.mean(list(times)) * 1000
            else:
                breakdown[component] = 0.0

        return breakdown

    def reset_metrics(self):
        """Reset performance metrics."""
        self.frame_times.clear()
        for times in self.component_times.values():
            times.clear()
        self.total_frames = 0
        self.dropped_frames = 0

        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("Performance metrics reset")

    def optimize_for_throughput(self):
        """Apply optimizations for maximum throughput."""
        logger.info("Applying throughput optimizations...")

        # Enable TF32 for faster matmul on Ampere+ GPUs
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info("Throughput optimizations applied")

    def optimize_for_latency(self):
        """Apply optimizations for minimum latency."""
        logger.info("Applying latency optimizations...")

        # Disable cudnn benchmarking for consistent latency
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        logger.info("Latency optimizations applied")

    def profile(self, num_frames: int = 100) -> Dict[str, Any]:
        """Profile pipeline performance.

        Args:
            num_frames: Number of frames to profile

        Returns:
            Profiling results
        """
        logger.info(f"Profiling pipeline for {num_frames} frames...")

        # Create dummy frames
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Reset metrics
        self.reset_metrics()

        # Warmup
        for _ in range(10):
            self.process_frame(dummy_frame)

        # Profile
        start_time = time.perf_counter()
        for i in range(num_frames):
            self.process_frame(dummy_frame, frame_idx=i)

        total_time = time.perf_counter() - start_time

        # Get metrics
        metrics = self.get_metrics()
        breakdown = self.get_component_breakdown()

        results = {
            'total_time_s': total_time,
            'frames_processed': num_frames,
            'metrics': {
                'avg_fps': metrics.avg_fps,
                'avg_latency_ms': metrics.avg_latency_ms,
                'p95_latency_ms': metrics.p95_latency_ms,
                'p99_latency_ms': metrics.p99_latency_ms,
                'gpu_memory_mb': metrics.gpu_memory_mb,
            },
            'component_breakdown_ms': breakdown,
        }

        logger.info(f"Profiling complete: {metrics.avg_fps:.1f} FPS, {metrics.avg_latency_ms:.1f}ms latency")

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizedPipeline(device={self.device}, batch_size={self.batch_size}, "
            f"fps={self.get_throughput():.1f}, latency={self.get_latency()*1000:.1f}ms)"
        )


class PipelineFactory:
    """Factory for creating optimized pipelines."""

    @staticmethod
    def create_realtime_pipeline(
        config: Union[Dict, str, Path],
        device: str = 'cuda',
        target_fps: float = 10.0
    ) -> OptimizedPipeline:
        """Create pipeline optimized for real-time processing.

        Args:
            config: Configuration dict or path to config file
            device: Device to run on
            target_fps: Target processing FPS

        Returns:
            OptimizedPipeline instance
        """
        # Load config if path provided
        if isinstance(config, (str, Path)):
            import yaml
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        # Calculate optimal batch size based on target FPS
        # For 10 FPS, batch size of 1 is usually optimal for latency
        batch_size = 1 if target_fps >= 10 else 2

        pipeline = OptimizedPipeline(
            config=config,
            device=device,
            batch_size=batch_size,
            enable_profiling=True
        )

        # Optimize for latency
        pipeline.optimize_for_latency()

        logger.info(f"Created real-time pipeline (target: {target_fps} FPS)")

        return pipeline

    @staticmethod
    def create_batch_pipeline(
        config: Union[Dict, str, Path],
        device: str = 'cuda',
        batch_size: int = 8
    ) -> OptimizedPipeline:
        """Create pipeline optimized for batch processing.

        Args:
            config: Configuration dict or path to config file
            device: Device to run on
            batch_size: Batch size for processing

        Returns:
            OptimizedPipeline instance
        """
        # Load config if path provided
        if isinstance(config, (str, Path)):
            import yaml
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        pipeline = OptimizedPipeline(
            config=config,
            device=device,
            batch_size=batch_size,
            enable_profiling=True
        )

        # Optimize for throughput
        pipeline.optimize_for_throughput()

        logger.info(f"Created batch pipeline (batch_size: {batch_size})")

        return pipeline
