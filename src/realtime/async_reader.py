"""Asynchronous video reading for real-time processing.

This module provides high-performance video reading with background buffering
to minimize I/O bottlenecks during real-time inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import time
from queue import Queue, Empty
from threading import Thread, Event
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent, Value
from ctypes import c_bool, c_double
import numpy as np
import cv2
from loguru import logger


@dataclass
class FrameData:
    """Container for frame data with metadata."""

    frame: np.ndarray
    timestamp: float
    frame_idx: int
    read_time: float = 0.0  # Time when frame was read
    is_valid: bool = True


class AsyncVideoReader:
    """Asynchronous video reader with background buffering.

    Reads frames in a background thread/process and maintains a buffer
    for smooth real-time processing. Supports both video files and live streams.
    """

    def __init__(
        self,
        source: str,
        buffer_size: int = 30,
        target_fps: Optional[float] = None,
        use_multiprocessing: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        skip_frames: int = 1
    ):
        """Initialize async video reader.

        Args:
            source: Video file path or stream URL (RTMP, HLS, etc.)
            buffer_size: Size of frame buffer
            target_fps: Target FPS for reading (None = read at source FPS)
            use_multiprocessing: Use multiprocessing instead of threading
            resize: Optional resize dimensions (width, height)
            skip_frames: Read every Nth frame (1 = all frames)
        """
        self.source = source
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.use_multiprocessing = use_multiprocessing
        self.resize = resize
        self.skip_frames = skip_frames

        # Video properties (populated on start)
        self.source_fps = None
        self.frame_count = None
        self.width = None
        self.height = None
        self.is_stream = self._check_if_stream(source)

        # Threading/multiprocessing components
        if use_multiprocessing:
            self.frame_queue = MPQueue(maxsize=buffer_size)
            self.stop_event = MPEvent()
            self.running = Value(c_bool, False)
            self.process = None
        else:
            self.frame_queue = Queue(maxsize=buffer_size)
            self.stop_event = Event()
            self.running = None
            self.thread = None

        # Statistics
        self.frames_read = 0
        self.frames_dropped = 0
        self.read_errors = 0
        self.start_time = None
        self.last_frame_time = 0.0

        logger.info(
            f"Initialized AsyncVideoReader: source={source}, "
            f"buffer_size={buffer_size}, is_stream={self.is_stream}, "
            f"multiprocessing={use_multiprocessing}"
        )

    def _check_if_stream(self, source: str) -> bool:
        """Check if source is a live stream."""
        stream_protocols = ['rtmp://', 'rtsp://', 'http://', 'https://']
        return any(source.startswith(proto) for proto in stream_protocols)

    def start(self):
        """Start background frame reading."""
        if self.use_multiprocessing:
            self.running.value = True
            self.process = Process(target=self._read_loop_mp, daemon=True)
            self.process.start()
            logger.info("Started async reader (multiprocessing)")
        else:
            self.thread = Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            logger.info("Started async reader (threading)")

        self.start_time = time.perf_counter()

        # Wait for first frame to populate video properties
        time.sleep(0.1)

    def stop(self):
        """Stop background frame reading."""
        if self.use_multiprocessing:
            self.running.value = False
            self.stop_event.set()
            if self.process is not None:
                self.process.join(timeout=2.0)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.0)
        else:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=2.0)

        logger.info(f"Stopped async reader. Read {self.frames_read} frames, dropped {self.frames_dropped}")

    def _read_loop(self):
        """Main reading loop (threading version)."""
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return

        # Get video properties
        self.source_fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.source_fps == 0:
            self.source_fps = 25.0  # Default for streams

        logger.info(
            f"Video source opened: {self.width}x{self.height} @ {self.source_fps} FPS"
        )

        # Calculate frame interval for target FPS
        if self.target_fps is not None and self.target_fps > 0:
            frame_interval = 1.0 / self.target_fps
        else:
            frame_interval = 1.0 / self.source_fps

        frame_idx = 0
        last_read_time = time.perf_counter()

        while not self.stop_event.is_set():
            current_time = time.perf_counter()

            # Rate limiting for target FPS
            if self.target_fps is not None:
                elapsed = current_time - last_read_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue

            # Read frame
            ret, frame = cap.read()

            if not ret:
                if self.is_stream:
                    # For streams, retry
                    logger.warning("Stream read failed, retrying...")
                    self.read_errors += 1
                    time.sleep(0.1)
                    continue
                else:
                    # For files, we're done
                    logger.info("End of video file reached")
                    break

            # Skip frames if configured
            if self.skip_frames > 1 and frame_idx % self.skip_frames != 0:
                frame_idx += 1
                continue

            # Resize if configured
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            # Calculate timestamp
            if self.is_stream:
                timestamp = current_time - self.start_time
            else:
                timestamp = frame_idx / self.source_fps

            # Create frame data
            frame_data = FrameData(
                frame=frame,
                timestamp=timestamp,
                frame_idx=frame_idx,
                read_time=current_time
            )

            # Try to put in queue (non-blocking)
            try:
                self.frame_queue.put(frame_data, block=False)
                self.frames_read += 1
                self.last_frame_time = current_time
            except:
                # Queue full, drop frame
                self.frames_dropped += 1

            frame_idx += 1
            last_read_time = current_time

        cap.release()
        logger.info("Read loop terminated")

    def _read_loop_mp(self):
        """Main reading loop (multiprocessing version)."""
        # Same as _read_loop but uses multiprocessing primitives
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return

        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps == 0:
            source_fps = 25.0

        frame_idx = 0
        start_time = time.perf_counter()

        if self.target_fps is not None and self.target_fps > 0:
            frame_interval = 1.0 / self.target_fps
        else:
            frame_interval = 1.0 / source_fps

        last_read_time = time.perf_counter()

        while self.running.value and not self.stop_event.is_set():
            current_time = time.perf_counter()

            # Rate limiting
            if self.target_fps is not None:
                elapsed = current_time - last_read_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue

            ret, frame = cap.read()

            if not ret:
                if self.is_stream:
                    time.sleep(0.1)
                    continue
                else:
                    break

            if self.skip_frames > 1 and frame_idx % self.skip_frames != 0:
                frame_idx += 1
                continue

            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            if self.is_stream:
                timestamp = current_time - start_time
            else:
                timestamp = frame_idx / source_fps

            frame_data = FrameData(
                frame=frame,
                timestamp=timestamp,
                frame_idx=frame_idx,
                read_time=current_time
            )

            try:
                self.frame_queue.put(frame_data, timeout=0.01)
            except:
                pass  # Drop frame if queue full

            frame_idx += 1
            last_read_time = current_time

        cap.release()

    def get_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        """Get next frame from buffer.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            FrameData object or None if timeout/stopped
        """
        try:
            frame_data = self.frame_queue.get(timeout=timeout)
            return frame_data
        except Empty:
            return None

    def is_live_stream(self) -> bool:
        """Check if source is a live stream."""
        return self.is_stream

    def get_latency(self) -> float:
        """Get current buffering latency in seconds.

        For streams: latency from frame capture to now.
        For files: processing lag.
        """
        if self.last_frame_time == 0:
            return 0.0

        current_time = time.perf_counter()
        return current_time - self.last_frame_time

    def get_buffer_fullness(self) -> float:
        """Get buffer fullness ratio (0.0 to 1.0)."""
        return self.frame_queue.qsize() / self.buffer_size

    def get_stats(self) -> dict:
        """Get reader statistics."""
        return {
            'frames_read': self.frames_read,
            'frames_dropped': self.frames_dropped,
            'read_errors': self.read_errors,
            'buffer_fullness': self.get_buffer_fullness(),
            'latency_s': self.get_latency(),
            'is_stream': self.is_stream,
            'source_fps': self.source_fps,
        }

    def seek(self, frame_idx: int) -> bool:
        """Seek to specific frame (only works for files, not streams).

        Args:
            frame_idx: Target frame index

        Returns:
            True if seek successful
        """
        if self.is_stream:
            logger.warning("Cannot seek in live streams")
            return False

        # Clear buffer
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        # Note: Seeking requires stopping and restarting with new position
        # This is a simplified version - full implementation would need more control
        logger.warning("Seeking requires reader restart (not fully implemented)")
        return False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class BatchFrameReader:
    """Read frames in batches for efficient processing.

    Wraps AsyncVideoReader to provide batched frame reading.
    """

    def __init__(
        self,
        source: str,
        batch_size: int = 8,
        buffer_size: int = 64,
        **kwargs
    ):
        """Initialize batch frame reader.

        Args:
            source: Video source
            batch_size: Number of frames per batch
            buffer_size: Frame buffer size (should be >= 2 * batch_size)
            **kwargs: Additional arguments for AsyncVideoReader
        """
        self.batch_size = batch_size
        self.reader = AsyncVideoReader(
            source=source,
            buffer_size=max(buffer_size, batch_size * 2),
            **kwargs
        )

    def start(self):
        """Start reading."""
        self.reader.start()

    def stop(self):
        """Stop reading."""
        self.reader.stop()

    def get_batch(self, timeout: float = 2.0) -> Optional[list]:
        """Get batch of frames.

        Args:
            timeout: Maximum time to wait for full batch

        Returns:
            List of FrameData objects (may be < batch_size at end)
        """
        batch = []
        start_time = time.perf_counter()

        while len(batch) < self.batch_size:
            remaining_time = timeout - (time.perf_counter() - start_time)
            if remaining_time <= 0:
                break

            frame_data = self.reader.get_frame(timeout=remaining_time)
            if frame_data is None:
                break

            batch.append(frame_data)

        return batch if batch else None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def benchmark_reader(
    source: str,
    duration: float = 10.0,
    use_multiprocessing: bool = False
) -> dict:
    """Benchmark video reader performance.

    Args:
        source: Video source to benchmark
        duration: Benchmark duration in seconds
        use_multiprocessing: Test multiprocessing mode

    Returns:
        Performance statistics
    """
    logger.info(f"Benchmarking reader for {duration}s...")

    with AsyncVideoReader(
        source=source,
        buffer_size=30,
        use_multiprocessing=use_multiprocessing
    ) as reader:
        start_time = time.perf_counter()
        frames_consumed = 0
        latencies = []

        while time.perf_counter() - start_time < duration:
            frame_data = reader.get_frame(timeout=1.0)
            if frame_data is None:
                break

            frames_consumed += 1
            latencies.append(reader.get_latency())

        stats = reader.get_stats()

    elapsed = time.perf_counter() - start_time
    fps = frames_consumed / elapsed

    results = {
        'duration_s': elapsed,
        'frames_consumed': frames_consumed,
        'fps': fps,
        'avg_latency_ms': np.mean(latencies) * 1000 if latencies else 0,
        'max_latency_ms': np.max(latencies) * 1000 if latencies else 0,
        **stats
    }

    logger.info(f"Reader benchmark: {fps:.1f} FPS, {results['avg_latency_ms']:.1f}ms latency")

    return results
