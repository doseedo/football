"""Main tracking pipeline."""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from .detection.player_detector import PlayerDetector
from .detection.ball_detector import BallDetector
from .detection.team_classifier import TeamClassifier, Team
from .tracking.tracker import PlayerTracker
from .homography.calibration import HomographyEstimator, InteractiveCalibrator, CoordinateTransformer
from .metrics.physical import PhysicalMetricsCalculator, KalmanSmoother
from .output.data_export import TrackingDataExporter, create_frame_record
from .output.visualizer import PitchVisualizer


class GreenPitchMaskDetector:
    def __init__(self, lower_green=(35, 50, 50), upper_green=(85, 255, 255)):
        self.lower_green = np.array(lower_green)
        self.upper_green = np.array(upper_green)

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask


class FootballTracker:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        pitch_template_path = Path(config_path).parent / "pitch_template.yaml"
        with open(pitch_template_path, 'r') as f:
            self.pitch_template = yaml.safe_load(f)

        self.device = self.config['processing']['device']
        self.fps = self.config['processing']['frame_rate']

        self._init_components()
        logger.info("Football tracker initialized")

    def _init_components(self):
        det_config = self.config['detection']
        self.player_detector = PlayerDetector(
            model_path=det_config['player']['model'],
            confidence=det_config['player']['confidence'],
            min_height=det_config['player']['min_height'],
            max_height=det_config['player']['max_height'],
            device=self.device
        )
        self.ball_detector = BallDetector(
            model_path=det_config['ball']['model'],
            confidence=det_config['ball']['confidence'],
            temporal_window=det_config['ball']['temporal_window'],
            device=self.device
        )

        team_config = self.config['team_classification']
        self.team_classifier = TeamClassifier(n_clusters=team_config['n_clusters'], color_space=team_config['color_space'])
        self.pitch_mask_detector = GreenPitchMaskDetector()

        track_config = self.config['tracking']
        self.player_tracker = PlayerTracker(
            track_thresh=track_config['track_high_thresh'],
            track_buffer=track_config['track_buffer'],
            match_thresh=track_config['match_thresh'],
            frame_rate=self.fps
        )

        self.interactive_calibrator = InteractiveCalibrator(self.pitch_template)
        self.transformer = CoordinateTransformer()

        self.metrics_calculator = PhysicalMetricsCalculator(fps=self.fps, smoothing_window=self.config['metrics']['speed_smoothing'])
        self.kalman_filters: Dict[int, KalmanSmoother] = {}
        self.position_history: Dict[int, List[tuple]] = {}

        self.exporter = TrackingDataExporter(self.config['video']['output_path'])
        self.visualizer = PitchVisualizer()

    def calibrate(self, video_path: str, interactive: bool = True):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        if interactive:
            logger.info("Starting interactive calibration...")
            result = self.interactive_calibrator.calibrate_interactive(frame)
        else:
            result = HomographyEstimator().estimate_from_manual_points([], [])

        if result.is_valid:
            self.transformer.set_homography(result.homography)
            logger.info(f"Calibration successful. Error: {result.reprojection_error:.2f}m")
        else:
            logger.error("Calibration failed!")

    def process_video(self, video_path: str, output_name: Optional[str] = None,
                     start_frame: int = 0, end_frame: Optional[int] = None, visualize: bool = False) -> List[Dict]:
        video_path = Path(video_path)
        output_name = output_name or video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames

        logger.info(f"Processing {video_path.name}: frames {start_frame}-{end_frame}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        if self.transformer.H is None:
            self.calibrate(str(video_path), interactive=True)

        pitch_mask = self.pitch_mask_detector.get_mask(first_frame)
        initial_detections = self.player_detector.detect(first_frame, pitch_mask)
        if len(initial_detections) >= 6:
            bboxes = [d.bbox for d in initial_detections]
            self.team_classifier.fit(first_frame, bboxes)
            logger.info("Team classifier fitted")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_records = []
        pbar = tqdm(total=end_frame - start_frame, desc="Processing")

        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            record = self._process_frame(frame, frame_idx)
            frame_records.append(record)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        metrics = self._calculate_final_metrics()
        self.exporter.export_frame_data(frame_records, output_name, formats=self.config['output']['format'])
        self.exporter.export_metrics(metrics, output_name)

        if visualize:
            self._generate_visualizations(frame_records, output_name)

        logger.info(f"Processing complete. {len(frame_records)} frames processed.")
        return frame_records

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        timestamp = frame_idx / self.fps
        pitch_mask = self.pitch_mask_detector.get_mask(frame)
        player_detections = self.player_detector.detect(frame, pitch_mask)
        player_bboxes = [d.bbox for d in player_detections]
        ball_detection = self.ball_detector.detect(frame, player_bboxes)
        tracks = self.player_tracker.update(player_detections, frame)

        player_records = []
        for track in tracks:
            pixel_pos = track.foot_position
            world_pos = self.transformer.pixel_to_world(pixel_pos)

            if track.track_id not in self.kalman_filters:
                self.kalman_filters[track.track_id] = KalmanSmoother(dt=1/self.fps)
            smoothed_pos = self.kalman_filters[track.track_id].update(world_pos)

            if track.track_id not in self.position_history:
                self.position_history[track.track_id] = []
            self.position_history[track.track_id].append(smoothed_pos)

            team_result = self.team_classifier.classify(frame, track.bbox)
            team_name = team_result.team.name.lower()

            positions = self.position_history[track.track_id]
            speed, accel = 0, 0
            if len(positions) >= 2:
                frame_metrics = self.metrics_calculator.calculate_frame_metrics(positions[-min(10, len(positions)):])
                if frame_metrics:
                    speed, accel = frame_metrics[-1].speed, frame_metrics[-1].acceleration

            player_records.append({
                'track_id': track.track_id, 'team': team_name,
                'jersey_number': None,
                'x': round(smoothed_pos[0], 2), 'y': round(smoothed_pos[1], 2),
                'speed': round(speed, 1), 'acceleration': round(accel, 2)
            })

        ball_record = None
        if ball_detection:
            ball_world = self.transformer.pixel_to_world(ball_detection.position)
            ball_record = {'x': round(ball_world[0], 2), 'y': round(ball_world[1], 2),
                          'is_interpolated': ball_detection.is_interpolated}

        return create_frame_record(frame_idx, timestamp, player_records, ball_record)

    def _calculate_final_metrics(self) -> List[Dict]:
        metrics = []
        for track_id, positions in self.position_history.items():
            if len(positions) < 10:
                continue
            track_metrics = self.metrics_calculator.calculate_match_metrics(positions, track_id)
            metrics.append({
                'track_id': track_id,
                'total_distance_m': round(track_metrics.total_distance, 1),
                'max_speed_kmh': round(track_metrics.max_speed, 1),
                'avg_speed_kmh': round(track_metrics.avg_speed, 1),
                'sprint_count': track_metrics.sprint_count,
                'sprint_distance_m': round(track_metrics.sprint_distance, 1),
                'high_intensity_distance_m': round(track_metrics.high_intensity_distance, 1),
                'max_acceleration_ms2': round(track_metrics.max_acceleration, 2),
                'max_deceleration_ms2': round(track_metrics.max_deceleration, 2)
            })
        return metrics

    def _generate_visualizations(self, frame_records: List[Dict], output_name: str):
        output_dir = Path(self.config['video']['output_path'])
        sample_indices = [0, len(frame_records)//4, len(frame_records)//2, 3*len(frame_records)//4, len(frame_records)-1]
        for idx in sample_indices:
            if idx < len(frame_records):
                record = frame_records[idx]
                self.visualizer.plot_frame(
                    players=record['players'], ball=record.get('ball'),
                    title=f"Frame {record['frame']}",
                    save_path=str(output_dir / f"{output_name}_frame_{idx}.png")
                )

        animation_records = frame_records[::5] if len(frame_records) > 100 else frame_records
        self.visualizer.create_animation(animation_records, str(output_dir / f"{output_name}_animation.mp4"), fps=5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Football XY Tracking")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--config", default="config/config.yaml", help="Config file")
    parser.add_argument("--output", help="Output name")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, help="End frame")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()

    tracker = FootballTracker(args.config)
    tracker.process_video(args.video, output_name=args.output, start_frame=args.start,
                         end_frame=args.end, visualize=args.visualize)


if __name__ == "__main__":
    main()
