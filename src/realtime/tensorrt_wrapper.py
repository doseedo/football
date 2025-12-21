"""TensorRT and ONNX optimization wrapper for model acceleration.

This module provides utilities to optimize PyTorch models using TensorRT and ONNX
for real-time inference performance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


# TensorRT imports (optional)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT not available. Install tensorrt for optimal performance.")

# ONNX imports (optional)
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install onnx and onnxruntime for fallback.")


class HostDeviceMem:
    """Simple helper data class to manage host/device memory allocations."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __repr__(self):
        return f"HostDeviceMem(host={self.host.nbytes} bytes, device={self.device})"


class TensorRTModel:
    """TensorRT optimized model wrapper with automatic fallback to PyTorch.

    Provides optimized inference using TensorRT with FP16/INT8 quantization.
    Falls back to PyTorch if TensorRT is not available or conversion fails.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        precision: str = 'fp16',
        max_batch_size: int = 16,
        device: str = 'cuda',
        workspace_size: int = 1 << 30  # 1GB
    ):
        """Initialize TensorRT wrapper.

        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape (B, C, H, W)
            precision: Precision mode ('fp32', 'fp16', 'int8')
            max_batch_size: Maximum batch size for inference
            device: Device to run on
            workspace_size: Maximum workspace size for TensorRT
        """
        self.model = model
        self.input_shape = input_shape
        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        self.device = device
        self.workspace_size = workspace_size

        # TensorRT engine
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        # Fallback to PyTorch
        self.use_pytorch = not TRT_AVAILABLE
        if self.use_pytorch and model is not None:
            self.model = model.to(device)
            self.model.eval()
            if precision == 'fp16' and device == 'cuda':
                self.model = self.model.half()

        # Performance metrics
        self.inference_times = []

    def convert_to_trt(
        self,
        calibration_data: Optional[List[np.ndarray]] = None,
        verbose: bool = False
    ) -> bool:
        """Convert PyTorch model to TensorRT engine.

        Args:
            calibration_data: Data for INT8 calibration (if precision='int8')
            verbose: Enable verbose TensorRT logging

        Returns:
            True if conversion successful, False otherwise
        """
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available, using PyTorch fallback")
            self.use_pytorch = True
            return False

        if self.model is None or self.input_shape is None:
            logger.error("Model and input_shape required for TensorRT conversion")
            return False

        try:
            logger.info(f"Converting model to TensorRT ({self.precision})...")

            # First export to ONNX
            onnx_path = Path("/tmp/model_temp.onnx")
            self._export_to_onnx(self.model, onnx_path, self.input_shape)

            # Build TensorRT engine from ONNX
            self._build_engine_from_onnx(onnx_path, calibration_data, verbose)

            # Clean up temp file
            if onnx_path.exists():
                onnx_path.unlink()

            logger.info("TensorRT conversion successful")
            self.use_pytorch = False
            return True

        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            logger.info("Falling back to PyTorch")
            self.use_pytorch = True
            return False

    def _export_to_onnx(
        self,
        model: nn.Module,
        path: Path,
        input_shape: Tuple[int, ...]
    ):
        """Export PyTorch model to ONNX."""
        model.eval()
        dummy_input = torch.randn(*input_shape).to(self.device)

        if self.precision == 'fp16' and self.device == 'cuda':
            dummy_input = dummy_input.half()

        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13,
            do_constant_folding=True
        )

    def _build_engine_from_onnx(
        self,
        onnx_path: Path,
        calibration_data: Optional[List[np.ndarray]],
        verbose: bool
    ):
        """Build TensorRT engine from ONNX model."""
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX file")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

        # Set precision
        if self.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data is not None:
                # Note: INT8 calibration requires custom calibrator
                logger.warning("INT8 calibration not implemented, using FP16 instead")
                config.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        logger.info("Building TensorRT engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Deserialize engine
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_buffers()

        # Create CUDA stream
        self.stream = cuda.Stream()

    def _allocate_buffers(self):
        """Allocate host and device buffers for inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)

            # Replace dynamic dimensions with max batch size
            shape = tuple(self.max_batch_size if dim == -1 else dim for dim in shape)
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to the appropriate list
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def infer(self, inputs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Run inference on input tensor.

        Args:
            inputs: Input tensor (numpy array or torch tensor)

        Returns:
            Output tensor in same format as input
        """
        start_time = time.perf_counter()

        # Convert to numpy if needed
        is_torch = isinstance(inputs, torch.Tensor)
        if is_torch:
            device_orig = inputs.device
            inputs_np = inputs.cpu().numpy()
        else:
            inputs_np = inputs

        # Perform inference
        if self.use_pytorch:
            outputs_np = self._infer_pytorch(inputs_np, is_torch)
        else:
            outputs_np = self._infer_tensorrt(inputs_np)

        # Convert back to torch if needed
        if is_torch:
            outputs = torch.from_numpy(outputs_np).to(device_orig)
        else:
            outputs = outputs_np

        # Track performance
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)

        return outputs

    def _infer_pytorch(self, inputs: np.ndarray, is_torch: bool) -> np.ndarray:
        """Fallback inference using PyTorch."""
        with torch.no_grad():
            inputs_torch = torch.from_numpy(inputs).to(self.device)
            if self.precision == 'fp16' and self.device == 'cuda':
                inputs_torch = inputs_torch.half()
            outputs_torch = self.model(inputs_torch)
            if isinstance(outputs_torch, (tuple, list)):
                outputs_torch = outputs_torch[0]
            return outputs_torch.cpu().numpy()

    def _infer_tensorrt(self, inputs: np.ndarray) -> np.ndarray:
        """TensorRT inference."""
        batch_size = inputs.shape[0]

        # Copy input to host buffer
        np.copyto(self.inputs[0].host, inputs.ravel())

        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # Set binding shapes
        self.context.set_input_shape("input", inputs.shape)

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back from device
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        # Reshape output
        output_shape = self.context.get_tensor_shape("output")
        output_shape = tuple(batch_size if dim == -1 else dim for dim in output_shape)

        return self.outputs[0].host[:np.prod(output_shape)].reshape(output_shape)

    def get_throughput(self) -> float:
        """Get average throughput in FPS."""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times[-100:])  # Last 100 inferences
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_latency(self) -> float:
        """Get average latency in seconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times[-100:])

    def save(self, path: Union[str, Path]):
        """Save TensorRT engine to file."""
        path = Path(path)

        if self.use_pytorch:
            # Save PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_shape': self.input_shape,
                'precision': self.precision,
            }, path)
            logger.info(f"Saved PyTorch model to {path}")
        else:
            # Save TensorRT engine
            with open(path, 'wb') as f:
                f.write(self.engine.serialize())
            logger.info(f"Saved TensorRT engine to {path}")

    def load(self, path: Union[str, Path]):
        """Load TensorRT engine or PyTorch model from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Try loading as TensorRT engine
        if TRT_AVAILABLE and path.suffix in ['.trt', '.engine']:
            try:
                self._load_tensorrt_engine(path)
                self.use_pytorch = False
                logger.info(f"Loaded TensorRT engine from {path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load TensorRT engine: {e}")

        # Fall back to PyTorch
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            raise ValueError("Model architecture required to load PyTorch weights")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_shape = checkpoint.get('input_shape', self.input_shape)
        self.precision = checkpoint.get('precision', self.precision)
        self.use_pytorch = True
        logger.info(f"Loaded PyTorch model from {path}")

    def _load_tensorrt_engine(self, path: Path):
        """Load serialized TensorRT engine."""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        with open(path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        self.stream = cuda.Stream()


class ONNXModel:
    """ONNX Runtime model wrapper for cross-platform deployment."""

    def __init__(
        self,
        onnx_path: Union[str, Path],
        device: str = 'cuda',
        providers: Optional[List[str]] = None
    ):
        """Initialize ONNX model.

        Args:
            onnx_path: Path to ONNX model file
            device: Device to run on ('cuda' or 'cpu')
            providers: ONNX Runtime execution providers
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")

        self.onnx_path = Path(onnx_path)
        self.device = device

        # Set providers
        if providers is None:
            if device == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Performance tracking
        self.inference_times = []

        logger.info(f"Loaded ONNX model from {onnx_path} with providers: {providers}")

    def infer(self, inputs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Run inference on input tensor.

        Args:
            inputs: Input tensor (numpy array or torch tensor)

        Returns:
            Output tensor in same format as input
        """
        start_time = time.perf_counter()

        # Convert to numpy if needed
        is_torch = isinstance(inputs, torch.Tensor)
        if is_torch:
            device_orig = inputs.device
            inputs_np = inputs.cpu().numpy()
        else:
            inputs_np = inputs

        # Run inference
        outputs_np = self.session.run(
            [self.output_name],
            {self.input_name: inputs_np}
        )[0]

        # Convert back to torch if needed
        if is_torch:
            outputs = torch.from_numpy(outputs_np).to(device_orig)
        else:
            outputs = outputs_np

        # Track performance
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)

        return outputs

    def get_throughput(self) -> float:
        """Get average throughput in FPS."""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times[-100:])
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_latency(self) -> float:
        """Get average latency in seconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times[-100:])


def export_onnx(
    model: nn.Module,
    path: Union[str, Path],
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    opset_version: int = 13,
    dynamic_batch: bool = True
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        path: Output path for ONNX file
        input_shape: Input tensor shape (B, C, H, W)
        device: Device for model
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size

    Returns:
        Path to exported ONNX file
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )

    # Verify ONNX model
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    logger.info(f"Exported ONNX model to {path}")
    return path


def load_onnx_runtime(
    path: Union[str, Path],
    device: str = 'cuda',
    providers: Optional[List[str]] = None
) -> ONNXModel:
    """Load ONNX model with ONNX Runtime.

    Args:
        path: Path to ONNX model file
        device: Device to run on
        providers: ONNX Runtime execution providers

    Returns:
        ONNXModel instance
    """
    return ONNXModel(path, device=device, providers=providers)


def benchmark_model(
    model: Union[TensorRTModel, ONNXModel, nn.Module],
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Benchmark model performance.

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of inference iterations
        warmup_iterations: Number of warmup iterations
        device: Device to run on

    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Benchmarking model with input shape {input_shape}...")

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(warmup_iterations):
        if isinstance(model, nn.Module):
            with torch.no_grad():
                _ = model(dummy_input)
        else:
            _ = model.infer(dummy_input)

    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(num_iterations):
        if isinstance(model, nn.Module):
            with torch.no_grad():
                _ = model(dummy_input)
        else:
            _ = model.infer(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    latency = avg_time * 1000  # ms

    metrics = {
        'avg_latency_ms': latency,
        'throughput_fps': throughput,
        'total_time_s': total_time,
        'num_iterations': num_iterations
    }

    logger.info(f"Benchmark results: {latency:.2f}ms latency, {throughput:.1f} FPS")

    return metrics
