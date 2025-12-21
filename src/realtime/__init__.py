"""Real-time pipeline optimization module.

This module provides optimized components for real-time football tracking:
- TensorRT and ONNX model optimization
- Asynchronous video reading with buffering
- Optimized inference pipeline with batching
- Live stream processing with backpressure handling

Performance targets:
- 10 FPS processing
- <2s end-to-end latency
- <4GB GPU memory usage

Example usage:
    >>> from realtime import PipelineFactory, StreamProcessor
    >>>
    >>> # Create optimized pipeline
    >>> pipeline = PipelineFactory.create_realtime_pipeline(
    ...     config='config/config.yaml',
    ...     target_fps=10.0
    ... )
    >>>
    >>> # Process live stream
    >>> processor = StreamProcessor(pipeline, source='rtmp://stream.url')
    >>> processor.start(callback_fn=lambda result: print(f"Frame {result.frame_idx}"))
    >>>
    >>> # Stop when done
    >>> processor.stop()
"""

# TensorRT and ONNX optimization
from .tensorrt_wrapper import (
    TensorRTModel,
    ONNXModel,
    export_onnx,
    load_onnx_runtime,
    benchmark_model,
    TRT_AVAILABLE,
    ONNX_AVAILABLE,
)

# Async video reading
from .async_reader import (
    AsyncVideoReader,
    BatchFrameReader,
    FrameData,
    benchmark_reader,
)

# Optimized pipeline
from .pipeline import (
    OptimizedPipeline,
    PipelineFactory,
    PipelineResult,
    PerformanceMetrics,
)

# Stream processing
from .stream_processor import (
    StreamProcessor,
    AdaptiveStreamProcessor,
    ProcessingStats,
    process_stream_to_callback,
    benchmark_stream_processor,
)

__all__ = [
    # TensorRT/ONNX
    'TensorRTModel',
    'ONNXModel',
    'export_onnx',
    'load_onnx_runtime',
    'benchmark_model',
    'TRT_AVAILABLE',
    'ONNX_AVAILABLE',
    # Async reading
    'AsyncVideoReader',
    'BatchFrameReader',
    'FrameData',
    'benchmark_reader',
    # Pipeline
    'OptimizedPipeline',
    'PipelineFactory',
    'PipelineResult',
    'PerformanceMetrics',
    # Stream processing
    'StreamProcessor',
    'AdaptiveStreamProcessor',
    'ProcessingStats',
    'process_stream_to_callback',
    'benchmark_stream_processor',
]

__version__ = '0.1.0'
