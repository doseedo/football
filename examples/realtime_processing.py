"""Example: Real-time Football Tracking

This script demonstrates how to use the real-time pipeline optimization module
to process football videos at 10 FPS with <2s latency.
"""

import sys
from pathlib import Path
import time
import argparse
import yaml
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from realtime import (
    PipelineFactory,
    StreamProcessor,
    AdaptiveStreamProcessor,
    benchmark_stream_processor,
)


def example_basic_realtime():
    """Example 1: Basic real-time processing with callbacks."""
    logger.info("Example 1: Basic real-time processing")

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    # Create optimized pipeline for real-time (10 FPS target)
    pipeline = PipelineFactory.create_realtime_pipeline(
        config=config_path,
        device='cuda',
        target_fps=10.0
    )

    # Set homography (in real use, calibrate first)
    # pipeline.set_homography(H_matrix)

    # Define callback for results
    def on_result(result):
        logger.info(
            f"Frame {result.frame_idx}: "
            f"{len(result.tracks)} players, "
            f"{result.processing_time*1000:.1f}ms"
        )

    # Create stream processor
    video_path = "path/to/video.mp4"  # Replace with actual video
    processor = StreamProcessor(
        pipeline=pipeline,
        source=video_path,
        target_fps=10.0,
        max_processing_latency=2.0,
        enable_frame_dropping=True
    )

    # Start processing
    processor.start(callback_fn=on_result)
    processor.log_stats_periodic(interval=5.0)

    # Run for 30 seconds
    try:
        time.sleep(30.0)
    except KeyboardInterrupt:
        logger.info("Interrupted")

    # Stop and show stats
    processor.stop()
    stats = processor.get_stats()

    logger.info(f"Processed {stats.frames_processed} frames at {stats.avg_fps:.1f} FPS")
    logger.info(f"Average latency: {stats.avg_latency_ms:.1f}ms")
    logger.info(f"End-to-end latency: {stats.end_to_end_latency_ms:.1f}ms")


def example_adaptive_fps():
    """Example 2: Adaptive FPS processing."""
    logger.info("Example 2: Adaptive FPS processing")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    pipeline = PipelineFactory.create_realtime_pipeline(
        config=config_path,
        device='cuda',
        target_fps=10.0
    )

    video_path = "path/to/video.mp4"

    # Use adaptive processor that automatically adjusts FPS
    processor = AdaptiveStreamProcessor(
        pipeline=pipeline,
        source=video_path,
        initial_fps=10.0,
        min_fps=5.0,
        max_fps=15.0,
        target_latency_ms=100.0,  # Target 100ms processing time
        adaptation_interval=5.0   # Adapt every 5 seconds
    )

    def on_result(result):
        logger.info(f"Frame {result.frame_idx}: {len(result.tracks)} players")

    processor.start(callback_fn=on_result)
    processor.log_stats_periodic(interval=5.0)

    try:
        time.sleep(60.0)  # Run for 1 minute
    except KeyboardInterrupt:
        pass

    processor.stop()


def example_live_stream():
    """Example 3: Live stream processing (RTMP/HLS)."""
    logger.info("Example 3: Live stream processing")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    pipeline = PipelineFactory.create_realtime_pipeline(
        config=config_path,
        device='cuda',
        target_fps=10.0
    )

    # Live stream URL (replace with actual stream)
    stream_url = "rtmp://live.stream.url/stream"

    processor = StreamProcessor(
        pipeline=pipeline,
        source=stream_url,
        target_fps=10.0,
        buffer_size=30,
        max_processing_latency=2.0,
        enable_frame_dropping=True,
        use_multiprocessing=True  # Use multiprocessing for live streams
    )

    results_buffer = []

    def on_result(result):
        results_buffer.append(result)
        # Save results, update UI, etc.
        if len(results_buffer) >= 10:
            # Process batch of results
            logger.info(f"Processed batch of {len(results_buffer)} frames")
            results_buffer.clear()

    processor.start(callback_fn=on_result)
    processor.log_stats_periodic(interval=10.0)

    try:
        # Run until interrupted
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping stream processing")

    processor.stop()


def example_batch_processing():
    """Example 4: Batch processing for maximum throughput."""
    logger.info("Example 4: Batch processing mode")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    # Create pipeline optimized for throughput
    pipeline = PipelineFactory.create_batch_pipeline(
        config=config_path,
        device='cuda',
        batch_size=8  # Process 8 frames at once
    )

    # Use async reader for efficient reading
    from realtime import BatchFrameReader

    video_path = "path/to/video.mp4"

    with BatchFrameReader(
        source=video_path,
        batch_size=8,
        buffer_size=64
    ) as reader:
        total_frames = 0

        while True:
            # Get batch of frames
            batch = reader.get_batch(timeout=2.0)
            if batch is None:
                break

            # Extract frames and metadata
            frames = [fd.frame for fd in batch]
            frame_indices = [fd.frame_idx for fd in batch]
            timestamps = [fd.timestamp for fd in batch]

            # Process batch
            results = pipeline.process_batch(frames, frame_indices, timestamps)

            total_frames += len(results)

            if total_frames % 100 == 0:
                metrics = pipeline.get_metrics()
                logger.info(
                    f"Processed {total_frames} frames at {metrics.avg_fps:.1f} FPS"
                )

    logger.info(f"Total frames processed: {total_frames}")


def example_tensorrt_optimization():
    """Example 5: TensorRT model optimization."""
    logger.info("Example 5: TensorRT optimization")

    from realtime import TensorRTModel, export_onnx
    import torch
    import torch.nn as nn

    # Example: Optimize a detection model
    class DummyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 270 * 480, 1000)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = DummyDetector().cuda().eval()
    input_shape = (1, 3, 1080, 1920)

    # Option 1: Convert to TensorRT
    trt_model = TensorRTModel(
        model=model,
        input_shape=input_shape,
        precision='fp16',
        device='cuda'
    )

    success = trt_model.convert_to_trt()

    if success:
        logger.info("TensorRT conversion successful")

        # Run inference
        dummy_input = torch.randn(*input_shape).cuda()
        output = trt_model.infer(dummy_input)

        logger.info(f"TensorRT throughput: {trt_model.get_throughput():.1f} FPS")

        # Save engine
        trt_model.save("models/detector.trt")

    # Option 2: Export to ONNX
    onnx_path = export_onnx(
        model=model,
        path="models/detector.onnx",
        input_shape=input_shape,
        device='cuda'
    )

    logger.info(f"Exported ONNX model to {onnx_path}")

    # Load and run ONNX model
    from realtime import load_onnx_runtime

    onnx_model = load_onnx_runtime(onnx_path, device='cuda')
    output = onnx_model.infer(dummy_input)

    logger.info(f"ONNX throughput: {onnx_model.get_throughput():.1f} FPS")


def run_benchmark():
    """Run comprehensive benchmark."""
    logger.info("Running comprehensive benchmark")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    video_path = "path/to/test_video.mp4"

    results = benchmark_stream_processor(
        config=config_path,
        source=video_path,
        duration=30.0,
        target_fps=10.0,
        device='cuda'
    )

    logger.info("Benchmark results:")
    logger.info(f"  Target FPS: {results['target_fps']}")
    logger.info(f"  Achieved FPS: {results['stream_stats']['avg_fps']:.1f}")
    logger.info(f"  Avg Latency: {results['stream_stats']['avg_latency_ms']:.1f}ms")
    logger.info(f"  E2E Latency: {results['stream_stats']['e2e_latency_ms']:.1f}ms")
    logger.info(f"  GPU Memory: {results['pipeline_stats']['gpu_memory_mb']:.1f} MB")

    # Check if targets are met
    targets_met = (
        results['stream_stats']['avg_fps'] >= 9.0 and  # At least 9 FPS
        results['stream_stats']['e2e_latency_ms'] < 2000 and  # <2s latency
        results['pipeline_stats']['gpu_memory_mb'] < 4096  # <4GB memory
    )

    if targets_met:
        logger.info("✓ All performance targets met!")
    else:
        logger.warning("✗ Some performance targets not met")


def main():
    parser = argparse.ArgumentParser(description="Real-time processing examples")
    parser.add_argument(
        '--example',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='Example to run (1-5)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark instead of examples'
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        examples = {
            1: example_basic_realtime,
            2: example_adaptive_fps,
            3: example_live_stream,
            4: example_batch_processing,
            5: example_tensorrt_optimization,
        }
        examples[args.example]()


if __name__ == "__main__":
    main()
