# Real-Time Pipeline Quick Start Guide

Get started with real-time football tracking in 5 minutes.

## Installation

```bash
# 1. Install base dependencies (if not already installed)
pip install torch torchvision opencv-python numpy loguru pyyaml

# 2. (Optional) Install acceleration libraries for maximum performance
# TensorRT (Linux/CUDA only - 2-5x speedup)
pip install tensorrt pycuda

# ONNX Runtime (Cross-platform)
pip install onnx onnxruntime-gpu
```

## Basic Usage

### 1. Real-Time Processing (10 FPS)

```python
from realtime import PipelineFactory, StreamProcessor

# Create optimized pipeline
pipeline = PipelineFactory.create_realtime_pipeline(
    config='config/config.yaml',
    target_fps=10.0
)

# Process video file
processor = StreamProcessor(
    pipeline=pipeline,
    source='video.mp4',
    target_fps=10.0
)

# Define callback for results
def on_result(result):
    print(f"Frame {result.frame_idx}: {len(result.tracks)} players tracked")
    # Save to database, update UI, etc.

# Start processing
processor.start(callback_fn=on_result)

# Run for 30 seconds or until video ends
import time
time.sleep(30)

# Stop and view stats
processor.stop()
stats = processor.get_stats()
print(f"Processed {stats.frames_processed} frames at {stats.avg_fps:.1f} FPS")
```

### 2. Live Stream Processing

```python
# Process RTMP/HLS live stream
processor = StreamProcessor(
    pipeline=pipeline,
    source="rtmp://live.stream.url/stream",
    target_fps=10.0,
    use_multiprocessing=True  # Recommended for streams
)

processor.start(callback_fn=on_result)
processor.log_stats_periodic(interval=5.0)  # Log stats every 5s

# Runs until stopped
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    processor.stop()
```

### 3. Batch Processing (Maximum Throughput)

```python
from realtime import PipelineFactory, BatchFrameReader

# Create batch pipeline
pipeline = PipelineFactory.create_batch_pipeline(
    config='config/config.yaml',
    batch_size=8
)

# Read and process in batches
with BatchFrameReader(source='video.mp4', batch_size=8) as reader:
    while True:
        batch = reader.get_batch()
        if batch is None:
            break
        
        # Process batch
        frames = [fd.frame for fd in batch]
        results = pipeline.process_batch(frames)
        
        # Handle results...
```

## Performance Optimization

### Enable TensorRT (2-5x speedup)

```python
from realtime import TensorRTModel, export_onnx

# Option 1: Convert your model to TensorRT
trt_model = TensorRTModel(
    model=your_pytorch_model,
    input_shape=(1, 3, 1080, 1920),
    precision='fp16'
)
success = trt_model.convert_to_trt()

# Option 2: Export to ONNX
onnx_path = export_onnx(
    model=your_pytorch_model,
    path='model.onnx',
    input_shape=(1, 3, 1080, 1920)
)
```

### Adaptive FPS Mode

```python
from realtime import AdaptiveStreamProcessor

# Automatically adjusts FPS based on performance
processor = AdaptiveStreamProcessor(
    pipeline=pipeline,
    source='video.mp4',
    initial_fps=10.0,
    min_fps=5.0,
    max_fps=15.0,
    target_latency_ms=100.0  # Target 100ms processing time
)

processor.start(on_result)
```

## Benchmarking

```python
from realtime import benchmark_stream_processor

# Run 30-second benchmark
results = benchmark_stream_processor(
    config='config/config.yaml',
    source='test_video.mp4',
    duration=30.0,
    target_fps=10.0
)

print(f"Achieved FPS: {results['stream_stats']['avg_fps']:.1f}")
print(f"E2E Latency: {results['stream_stats']['e2e_latency_ms']:.1f}ms")
print(f"GPU Memory: {results['pipeline_stats']['gpu_memory_mb']:.1f} MB")
```

## Monitoring Performance

```python
# Get real-time stats
stats = processor.get_stats()
print(f"FPS: {stats.avg_fps:.1f}")
print(f"Latency: {stats.avg_latency_ms:.1f}ms")
print(f"Dropped: {stats.frames_dropped}")
print(f"Buffer: {stats.buffer_fullness*100:.0f}%")

# Get pipeline metrics
metrics = pipeline.get_metrics()
print(f"Avg FPS: {metrics.avg_fps:.1f}")
print(f"P95 Latency: {metrics.p95_latency_ms:.1f}ms")
print(f"GPU Memory: {metrics.gpu_memory_mb:.1f} MB")

# Component breakdown
breakdown = pipeline.get_component_breakdown()
for component, time_ms in breakdown.items():
    print(f"{component}: {time_ms:.1f}ms")
```

## Configuration Tips

### For Low Latency (<100ms)
- Use batch_size=1
- Enable frame dropping
- Set max_processing_latency=0.2
- Use FP16 precision

### For High Throughput
- Use larger batch sizes (4-8)
- Disable frame dropping
- Use batch processing mode
- Consider TensorRT optimization

### For Live Streams
- Use multiprocessing mode
- Enable adaptive FPS
- Set appropriate buffer size (30-60)
- Monitor end-to-end latency

## Troubleshooting

### Slow Performance (<10 FPS)
1. Enable TensorRT: `pip install tensorrt pycuda`
2. Use FP16 precision
3. Reduce input resolution
4. Check GPU utilization

### High Latency (>2s)
1. Enable frame dropping
2. Reduce buffer size
3. Use adaptive FPS mode
4. Check network latency (for streams)

### Out of Memory
1. Reduce batch size
2. Use smaller models
3. Lower input resolution
4. Enable gradient checkpointing

## Examples

See `/home/user/football/examples/realtime_processing.py` for:
- Example 1: Basic real-time processing
- Example 2: Adaptive FPS processing
- Example 3: Live stream processing
- Example 4: Batch processing
- Example 5: TensorRT optimization

Run examples:
```bash
python examples/realtime_processing.py --example 1
python examples/realtime_processing.py --benchmark
```

## Next Steps

1. Read the full documentation: `src/realtime/README.md`
2. Run the test suite: `python tests/test_realtime.py`
3. Try the examples: `python examples/realtime_processing.py`
4. Benchmark your setup: `--benchmark` flag
5. Optimize with TensorRT for maximum performance

---

For detailed API documentation, see `src/realtime/README.md`

For implementation details, see `REALTIME_IMPLEMENTATION.md`
