"""Live stream processing for real-time football tracking.

This module combines async video reading with optimized inference pipeline
to provide real-time processing with automatic frame dropping and backpressure handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import time
from threading import Thread, Event, Lock
from queue import Queue, Empty
import numpy as np
from loguru import logger

from .async_reader import AsyncVideoReader, FrameData
from .pipeline import OptimizedPipeline, PipelineResult


@dataclass
class ProcessingStats:
    """Statistics for stream processing."""

    frames_read: int
    frames_processed: int
    frames_dropped: int
    avg_fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    buffer_fullness: float
    is_running: bool
    end_to_end_latency_ms: float


class StreamProcessor:
    """Real-time stream processor with automatic backpressure handling.

    Combines AsyncVideoReader with OptimizedPipeline to provide:
    - Real-time processing at target FPS
    - Automatic frame dropping when falling behind
    - Callbacks for results
    - Performance monitoring
    """

    def __init__(
        self,
        pipeline: OptimizedPipeline,
        source: str,
        target_fps: float = 10.0,
        buffer_size: int = 30,
        max_processing_latency: float = 2.0,
        enable_frame_dropping: bool = True,
        use_multiprocessing: bool = False
    ):
        """Initialize stream processor.

        Args:
            pipeline: Optimized inference pipeline
            source: Video source (file or stream URL)
            target_fps: Target processing FPS
            buffer_size: Frame buffer size
            max_processing_latency: Maximum allowed latency before dropping frames
            enable_frame_dropping: Enable automatic frame dropping
            use_multiprocessing: Use multiprocessing for video reading
        """
        self.pipeline = pipeline
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.max_processing_latency = max_processing_latency
        self.enable_frame_dropping = enable_frame_dropping

        # Calculate frame skip for target FPS
        source_fps = 25.0  # Default, will be updated when reader starts
        self.frame_skip = max(1, int(source_fps / target_fps))

        # Initialize async reader
        self.reader = AsyncVideoReader(
            source=source,
            buffer_size=buffer_size,
            target_fps=target_fps,
            use_multiprocessing=use_multiprocessing,
            skip_frames=1  # Don't skip in reader, we'll handle it
        )

        # Processing thread
        self.processing_thread = None
        self.stop_event = Event()
        self.is_running = False

        # Result callback
        self.callback_fn = None
        self.result_queue = Queue(maxsize=100)

        # Statistics
        self.stats_lock = Lock()
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.processing_times = []
        self.e2e_latencies = []  # End-to-end latency
        self.last_processed_idx = -1

        logger.info(
            f"StreamProcessor initialized: source={source}, "
            f"target_fps={target_fps}, buffer_size={buffer_size}"
        )

    def start(self, callback_fn: Optional[Callable[[PipelineResult], None]] = None):
        """Start stream processing.

        Args:
            callback_fn: Optional callback function for results
        """
        if self.is_running:
            logger.warning("Stream processor already running")
            return

        self.callback_fn = callback_fn
        self.stop_event.clear()

        # Start reader
        self.reader.start()
        time.sleep(0.2)  # Give reader time to start

        # Update frame skip based on actual source FPS
        if self.reader.source_fps:
            self.frame_skip = max(1, int(self.reader.source_fps / self.target_fps))
            logger.info(f"Source FPS: {self.reader.source_fps}, frame_skip: {self.frame_skip}")

        # Start processing thread
        self.processing_thread = Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.is_running = True

        logger.info("Stream processing started")

    def stop(self):
        """Stop stream processing."""
        if not self.is_running:
            return

        logger.info("Stopping stream processor...")

        self.stop_event.set()
        self.is_running = False

        # Stop reader
        self.reader.stop()

        # Wait for processing thread
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=3.0)

        logger.info("Stream processor stopped")
        self._log_final_stats()

    def _processing_loop(self):
        """Main processing loop."""
        last_process_time = time.perf_counter()
        frame_interval = 1.0 / self.target_fps

        while not self.stop_event.is_set():
            current_time = time.perf_counter()

            # Rate limiting
            elapsed = current_time - last_process_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            # Get frame from reader
            frame_data = self.reader.get_frame(timeout=0.5)

            if frame_data is None:
                # No frame available
                continue

            with self.stats_lock:
                self.frames_read += 1

            # Check if we should process this frame
            should_process = self._should_process_frame(frame_data)

            if not should_process:
                with self.stats_lock:
                    self.frames_dropped += 1
                continue

            # Process frame
            try:
                start_time = time.perf_counter()

                result = self.pipeline.process_frame(
                    frame=frame_data.frame,
                    frame_idx=frame_data.frame_idx,
                    timestamp=frame_data.timestamp
                )

                processing_time = time.perf_counter() - start_time

                # Calculate end-to-end latency
                e2e_latency = time.perf_counter() - frame_data.read_time

                with self.stats_lock:
                    self.frames_processed += 1
                    self.processing_times.append(processing_time)
                    self.e2e_latencies.append(e2e_latency)
                    self.last_processed_idx = frame_data.frame_idx

                    # Keep only recent times
                    if len(self.processing_times) > 1000:
                        self.processing_times = self.processing_times[-1000:]
                    if len(self.e2e_latencies) > 1000:
                        self.e2e_latencies = self.e2e_latencies[-1000:]

                # Invoke callback
                if self.callback_fn is not None:
                    try:
                        self.callback_fn(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # Put in result queue
                try:
                    self.result_queue.put(result, block=False)
                except:
                    pass  # Queue full

                last_process_time = time.perf_counter()

            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue

    def _should_process_frame(self, frame_data: FrameData) -> bool:
        """Determine if frame should be processed.

        Args:
            frame_data: Frame data to check

        Returns:
            True if frame should be processed
        """
        # Skip frames based on target FPS
        if self.frame_skip > 1 and frame_data.frame_idx % self.frame_skip != 0:
            return False

        # Check for frame skipping due to backpressure
        if not self.enable_frame_dropping:
            return True

        # If we're falling behind, drop frames
        current_time = time.perf_counter()
        frame_age = current_time - frame_data.read_time

        if frame_age > self.max_processing_latency:
            logger.debug(f"Dropping frame {frame_data.frame_idx} (age: {frame_age:.3f}s)")
            return False

        # Check buffer fullness - if too full, we're falling behind
        buffer_fullness = self.reader.get_buffer_fullness()
        if buffer_fullness > 0.8:  # 80% full
            # Drop every other frame to catch up
            if frame_data.frame_idx % 2 != 0:
                return False

        return True

    def get_result(self, timeout: float = 1.0) -> Optional[PipelineResult]:
        """Get next processing result.

        Args:
            timeout: Maximum time to wait

        Returns:
            PipelineResult or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics.

        Returns:
            ProcessingStats object
        """
        with self.stats_lock:
            # Calculate metrics
            if self.processing_times:
                avg_latency = np.mean(self.processing_times) * 1000
                p95_latency = np.percentile(self.processing_times, 95) * 1000
            else:
                avg_latency = 0.0
                p95_latency = 0.0

            if self.e2e_latencies:
                e2e_latency = np.mean(self.e2e_latencies) * 1000
            else:
                e2e_latency = 0.0

            # Calculate FPS
            if self.processing_times and len(self.processing_times) > 1:
                avg_fps = 1.0 / np.mean(self.processing_times)
            else:
                avg_fps = 0.0

            return ProcessingStats(
                frames_read=self.frames_read,
                frames_processed=self.frames_processed,
                frames_dropped=self.frames_dropped,
                avg_fps=avg_fps,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                buffer_fullness=self.reader.get_buffer_fullness(),
                is_running=self.is_running,
                end_to_end_latency_ms=e2e_latency
            )

    def _log_final_stats(self):
        """Log final statistics."""
        stats = self.get_stats()

        logger.info("=" * 60)
        logger.info("Stream Processing Statistics:")
        logger.info(f"  Frames read:      {stats.frames_read}")
        logger.info(f"  Frames processed: {stats.frames_processed}")
        logger.info(f"  Frames dropped:   {stats.frames_dropped}")

        if stats.frames_read > 0:
            drop_rate = 100 * stats.frames_dropped / stats.frames_read
            logger.info(f"  Drop rate:        {drop_rate:.1f}%")

        logger.info(f"  Avg FPS:          {stats.avg_fps:.1f}")
        logger.info(f"  Avg latency:      {stats.avg_latency_ms:.1f}ms")
        logger.info(f"  P95 latency:      {stats.p95_latency_ms:.1f}ms")
        logger.info(f"  E2E latency:      {stats.end_to_end_latency_ms:.1f}ms")
        logger.info("=" * 60)

    def log_stats_periodic(self, interval: float = 10.0):
        """Start periodic stats logging.

        Args:
            interval: Logging interval in seconds
        """
        def _log_loop():
            while not self.stop_event.is_set():
                time.sleep(interval)
                if self.is_running:
                    stats = self.get_stats()
                    logger.info(
                        f"[Stream Stats] FPS: {stats.avg_fps:.1f}, "
                        f"Latency: {stats.avg_latency_ms:.1f}ms, "
                        f"E2E: {stats.end_to_end_latency_ms:.1f}ms, "
                        f"Dropped: {stats.frames_dropped}, "
                        f"Buffer: {stats.buffer_fullness*100:.0f}%"
                    )

        log_thread = Thread(target=_log_loop, daemon=True)
        log_thread.start()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class AdaptiveStreamProcessor(StreamProcessor):
    """Stream processor with adaptive FPS based on processing performance.

    Automatically adjusts target FPS to maintain latency targets.
    """

    def __init__(
        self,
        pipeline: OptimizedPipeline,
        source: str,
        initial_fps: float = 10.0,
        min_fps: float = 5.0,
        max_fps: float = 15.0,
        target_latency_ms: float = 100.0,
        adaptation_interval: float = 5.0,
        **kwargs
    ):
        """Initialize adaptive stream processor.

        Args:
            pipeline: Optimized pipeline
            source: Video source
            initial_fps: Initial target FPS
            min_fps: Minimum allowed FPS
            max_fps: Maximum allowed FPS
            target_latency_ms: Target processing latency
            adaptation_interval: How often to adapt FPS
            **kwargs: Additional arguments for StreamProcessor
        """
        super().__init__(
            pipeline=pipeline,
            source=source,
            target_fps=initial_fps,
            **kwargs
        )

        self.min_fps = min_fps
        self.max_fps = max_fps
        self.target_latency_ms = target_latency_ms
        self.adaptation_interval = adaptation_interval
        self.adaptation_thread = None

        logger.info(
            f"AdaptiveStreamProcessor: FPS range [{min_fps}, {max_fps}], "
            f"target latency {target_latency_ms}ms"
        )

    def start(self, callback_fn: Optional[Callable[[PipelineResult], None]] = None):
        """Start processing with adaptation."""
        super().start(callback_fn)

        # Start adaptation thread
        self.adaptation_thread = Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()

        logger.info("Adaptive stream processing started")

    def _adaptation_loop(self):
        """Periodically adapt FPS based on performance."""
        while not self.stop_event.is_set():
            time.sleep(self.adaptation_interval)

            if not self.is_running:
                continue

            stats = self.get_stats()

            # Adapt FPS based on latency
            if stats.avg_latency_ms > self.target_latency_ms * 1.2:
                # Slow down
                new_fps = max(self.min_fps, self.target_fps * 0.9)
                if new_fps != self.target_fps:
                    logger.info(
                        f"Reducing FPS: {self.target_fps:.1f} -> {new_fps:.1f} "
                        f"(latency: {stats.avg_latency_ms:.1f}ms)"
                    )
                    self.target_fps = new_fps
            elif stats.avg_latency_ms < self.target_latency_ms * 0.8:
                # Speed up
                new_fps = min(self.max_fps, self.target_fps * 1.1)
                if new_fps != self.target_fps:
                    logger.info(
                        f"Increasing FPS: {self.target_fps:.1f} -> {new_fps:.1f} "
                        f"(latency: {stats.avg_latency_ms:.1f}ms)"
                    )
                    self.target_fps = new_fps


def process_stream_to_callback(
    config: Union[Dict, str, Path],
    source: str,
    callback_fn: Callable[[PipelineResult], None],
    target_fps: float = 10.0,
    device: str = 'cuda',
    duration: Optional[float] = None
):
    """Convenience function to process stream with callback.

    Args:
        config: Pipeline configuration
        source: Video source
        callback_fn: Callback for results
        target_fps: Target processing FPS
        device: Device to run on
        duration: Optional duration to process (None = until end)
    """
    # Load config
    if isinstance(config, (str, Path)):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    # Create pipeline
    from .pipeline import PipelineFactory
    pipeline = PipelineFactory.create_realtime_pipeline(
        config=config,
        device=device,
        target_fps=target_fps
    )

    # Create processor
    processor = StreamProcessor(
        pipeline=pipeline,
        source=source,
        target_fps=target_fps
    )

    # Start processing
    processor.start(callback_fn)
    processor.log_stats_periodic(interval=5.0)

    # Run for duration or until interrupted
    try:
        if duration is not None:
            time.sleep(duration)
        else:
            # Run until KeyboardInterrupt
            while processor.is_running:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.stop()


def benchmark_stream_processor(
    config: Union[Dict, str, Path],
    source: str,
    duration: float = 30.0,
    target_fps: float = 10.0,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Benchmark stream processor performance.

    Args:
        config: Pipeline configuration
        source: Video source
        duration: Benchmark duration
        target_fps: Target FPS
        device: Device to run on

    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking stream processor for {duration}s...")

    # Load config
    if isinstance(config, (str, Path)):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    # Create pipeline
    from .pipeline import PipelineFactory
    pipeline = PipelineFactory.create_realtime_pipeline(
        config=config,
        device=device,
        target_fps=target_fps
    )

    # Create processor
    processor = StreamProcessor(
        pipeline=pipeline,
        source=source,
        target_fps=target_fps
    )

    # Run benchmark
    results = []

    def collect_result(result: PipelineResult):
        results.append(result)

    processor.start(collect_result)

    time.sleep(duration)

    processor.stop()

    # Get stats
    stats = processor.get_stats()
    pipeline_metrics = pipeline.get_metrics()

    benchmark_results = {
        'duration_s': duration,
        'target_fps': target_fps,
        'stream_stats': {
            'frames_read': stats.frames_read,
            'frames_processed': stats.frames_processed,
            'frames_dropped': stats.frames_dropped,
            'avg_fps': stats.avg_fps,
            'avg_latency_ms': stats.avg_latency_ms,
            'p95_latency_ms': stats.p95_latency_ms,
            'e2e_latency_ms': stats.end_to_end_latency_ms,
        },
        'pipeline_stats': {
            'avg_fps': pipeline_metrics.avg_fps,
            'avg_latency_ms': pipeline_metrics.avg_latency_ms,
            'p95_latency_ms': pipeline_metrics.p95_latency_ms,
            'p99_latency_ms': pipeline_metrics.p99_latency_ms,
            'gpu_memory_mb': pipeline_metrics.gpu_memory_mb,
        },
        'component_breakdown_ms': pipeline.get_component_breakdown(),
    }

    logger.info("=" * 60)
    logger.info("Benchmark Results:")
    logger.info(f"  Target FPS:       {target_fps}")
    logger.info(f"  Achieved FPS:     {stats.avg_fps:.1f}")
    logger.info(f"  Avg Latency:      {stats.avg_latency_ms:.1f}ms")
    logger.info(f"  E2E Latency:      {stats.end_to_end_latency_ms:.1f}ms")
    logger.info(f"  Frames Dropped:   {stats.frames_dropped} ({100*stats.frames_dropped/max(stats.frames_read, 1):.1f}%)")
    logger.info(f"  GPU Memory:       {pipeline_metrics.gpu_memory_mb:.1f} MB")
    logger.info("=" * 60)

    return benchmark_results
