"""
Model Optimizer Service
Optimizes models for deployment with TensorRT, quantization, and pruning
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import onnx
import tensorrt as trt
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
optimization_counter = Counter('model_optimizations_total', 'Total number of model optimizations')
optimization_duration = Histogram('model_optimization_duration_seconds', 'Model optimization duration')
optimization_errors = Counter('model_optimization_errors_total', 'Total number of optimization errors')
model_size_reduction = Gauge('model_size_reduction_percent', 'Model size reduction percentage')

app = FastAPI(title="Model Optimizer Service", version="1.0.0")

class OptimizationRequest(BaseModel):
    model_name: str
    model_path: str
    optimization_level: str = "O3"  # O1, O2, O3
    target_precision: str = "FP16"  # FP32, FP16, INT8
    enable_tensorrt: bool = True
    enable_quantization: bool = True
    enable_pruning: bool = False
    pruning_sparsity: float = 0.5
    batch_sizes: List[int] = [1, 8, 16, 32]
    max_workspace_size_gb: int = 4

class OptimizationResult(BaseModel):
    success: bool
    optimized_model_path: str
    original_size_mb: float
    optimized_size_mb: float
    size_reduction_percent: float
    optimization_time_seconds: float
    techniques_applied: List[str]
    performance_metrics: Dict[str, Any]

class ModelOptimizer:
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.redis_client = None
        self.cache_dir = Path("/cache")
        self.model_dir = Path("/models")

        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialize connections"""
        try:
            self.redis_client = await redis.Redis.from_url(
                "redis://redis-master-1:7000",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    def optimize_onnx_model(self, model_path: str, config: OptimizationRequest) -> Dict[str, Any]:
        """Optimize ONNX model"""
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType

        techniques = []
        metrics = {}

        # Load model
        model = onnx.load(model_path)
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

        # Graph optimization
        if config.optimization_level in ["O2", "O3"]:
            from onnx import optimizer
            optimized_model = optimizer.optimize(model)
            techniques.append("graph_optimization")
        else:
            optimized_model = model

        # Quantization
        if config.enable_quantization:
            output_path = self.cache_dir / f"{config.model_name}_quantized.onnx"

            if config.target_precision == "INT8":
                quantize_dynamic(
                    model_path,
                    str(output_path),
                    weight_type=QuantType.QInt8
                )
                techniques.append("int8_quantization")
            else:
                # FP16 conversion
                from onnxconverter_common import float16
                model_fp16 = float16.convert_float_to_float16(optimized_model)
                onnx.save(model_fp16, str(output_path))
                techniques.append("fp16_conversion")

            optimized_size = os.path.getsize(output_path) / (1024 * 1024)
        else:
            output_path = self.cache_dir / f"{config.model_name}_optimized.onnx"
            onnx.save(optimized_model, str(output_path))
            optimized_size = os.path.getsize(output_path) / (1024 * 1024)

        # Calculate metrics
        metrics["original_size_mb"] = original_size
        metrics["optimized_size_mb"] = optimized_size
        metrics["size_reduction_percent"] = ((original_size - optimized_size) / original_size) * 100

        return {
            "output_path": str(output_path),
            "techniques": techniques,
            "metrics": metrics
        }

    def optimize_with_tensorrt(self, onnx_path: str, config: OptimizationRequest) -> Dict[str, Any]:
        """Optimize model with TensorRT"""
        techniques = []
        metrics = {}

        # Create builder
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.trt_logger)

        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(parser.get_error(i))
                raise RuntimeError(f"TensorRT parsing errors: {errors}")

        # Configure builder
        config_trt = builder.create_builder_config()
        config_trt.max_workspace_size = config.max_workspace_size_gb * (1 << 30)  # GB to bytes

        # Set precision
        if config.target_precision == "FP16":
            config_trt.set_flag(trt.BuilderFlag.FP16)
            techniques.append("tensorrt_fp16")
        elif config.target_precision == "INT8":
            config_trt.set_flag(trt.BuilderFlag.INT8)
            techniques.append("tensorrt_int8")
            # Would need calibration data for INT8

        # Optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()

        # Add optimization profiles for different batch sizes
        for input_idx in range(network.num_inputs):
            input_tensor = network.get_input(input_idx)
            shape = input_tensor.shape

            # Set min, optimal, max shapes
            min_shape = [1 if i == 0 else s for i, s in enumerate(shape)]
            opt_shape = [16 if i == 0 else s for i, s in enumerate(shape)]
            max_shape = [config.batch_sizes[-1] if i == 0 else s for i, s in enumerate(shape)]

            profile.set_shape(
                input_tensor.name,
                min_shape,
                opt_shape,
                max_shape
            )

        config_trt.add_optimization_profile(profile)

        # Build engine
        logger.info(f"Building TensorRT engine for {config.model_name}")
        engine = builder.build_engine(network, config_trt)

        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Serialize engine
        output_path = self.cache_dir / f"{config.model_name}.trt"
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())

        # Calculate metrics
        engine_size = os.path.getsize(output_path) / (1024 * 1024)
        metrics["tensorrt_engine_size_mb"] = engine_size
        metrics["num_layers"] = network.num_layers
        metrics["workspace_size_mb"] = config.max_workspace_size_gb * 1024

        # Benchmark engine
        benchmark_results = self.benchmark_tensorrt_engine(engine, config.batch_sizes)
        metrics.update(benchmark_results)

        return {
            "output_path": str(output_path),
            "techniques": techniques,
            "metrics": metrics
        }

    def benchmark_tensorrt_engine(self, engine, batch_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark TensorRT engine performance"""
        import pycuda.driver as cuda
        import pycuda.autoinit

        metrics = {}
        context = engine.create_execution_context()

        for batch_size in batch_sizes:
            # Allocate buffers
            inputs = []
            outputs = []
            bindings = []
            stream = cuda.Stream()

            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                size = trt.volume(engine.get_binding_shape(binding_idx)) * batch_size
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                bindings.append(int(device_mem))

                if engine.binding_is_input(binding):
                    inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem})

            # Warm up
            for _ in range(10):
                context.execute_async_v2(bindings, stream.handle)
            stream.synchronize()

            # Benchmark
            num_iterations = 100
            start_time = time.perf_counter()

            for _ in range(num_iterations):
                # Transfer input data to GPU
                for inp in inputs:
                    cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

                # Run inference
                context.execute_async_v2(bindings, stream.handle)

                # Transfer predictions back
                for out in outputs:
                    cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

                stream.synchronize()

            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            avg_latency_ms = (total_time / num_iterations) * 1000
            throughput = (batch_size * num_iterations) / total_time

            metrics[f"latency_ms_batch_{batch_size}"] = avg_latency_ms
            metrics[f"throughput_samples_per_sec_batch_{batch_size}"] = throughput

        return metrics

    def apply_pruning(self, model_path: str, config: OptimizationRequest) -> Dict[str, Any]:
        """Apply model pruning"""
        import torch.nn.utils.prune as prune

        techniques = ["structured_pruning"]
        metrics = {}

        # Load PyTorch model (if applicable)
        # This is a simplified example
        model = torch.load(model_path)

        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters())

        # Apply structured pruning
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=config.pruning_sparsity
                )

        # Remove pruning reparameterization
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.remove(module, 'weight')

        # Count remaining parameters
        remaining_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Save pruned model
        output_path = self.cache_dir / f"{config.model_name}_pruned.pt"
        torch.save(model, output_path)

        metrics["original_parameters"] = original_params
        metrics["remaining_parameters"] = remaining_params
        metrics["sparsity_percent"] = ((original_params - remaining_params) / original_params) * 100

        return {
            "output_path": str(output_path),
            "techniques": techniques,
            "metrics": metrics
        }

    async def optimize_model(self, config: OptimizationRequest) -> OptimizationResult:
        """Main optimization pipeline"""
        start_time = time.time()
        techniques_applied = []
        all_metrics = {}

        try:
            # Check cache
            cache_key = f"optimized_model:{config.model_name}:{config.optimization_level}:{config.target_precision}"
            cached_result = await self.redis_client.get(cache_key) if self.redis_client else None

            if cached_result:
                logger.info(f"Using cached optimization for {config.model_name}")
                return OptimizationResult(**eval(cached_result))

            # ONNX optimization
            onnx_result = self.optimize_onnx_model(config.model_path, config)
            techniques_applied.extend(onnx_result["techniques"])
            all_metrics.update(onnx_result["metrics"])

            # TensorRT optimization
            if config.enable_tensorrt:
                trt_result = self.optimize_with_tensorrt(
                    onnx_result["output_path"],
                    config
                )
                techniques_applied.extend(trt_result["techniques"])
                all_metrics.update(trt_result["metrics"])
                final_path = trt_result["output_path"]
            else:
                final_path = onnx_result["output_path"]

            # Pruning (if applicable)
            if config.enable_pruning and config.model_path.endswith('.pt'):
                pruning_result = self.apply_pruning(config.model_path, config)
                techniques_applied.extend(pruning_result["techniques"])
                all_metrics.update(pruning_result["metrics"])

            # Calculate final metrics
            optimization_time = time.time() - start_time

            result = OptimizationResult(
                success=True,
                optimized_model_path=final_path,
                original_size_mb=all_metrics.get("original_size_mb", 0),
                optimized_size_mb=all_metrics.get("optimized_size_mb", 0),
                size_reduction_percent=all_metrics.get("size_reduction_percent", 0),
                optimization_time_seconds=optimization_time,
                techniques_applied=techniques_applied,
                performance_metrics=all_metrics
            )

            # Update metrics
            optimization_counter.inc()
            optimization_duration.observe(optimization_time)
            model_size_reduction.set(result.size_reduction_percent)

            # Cache result
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    str(result.dict())
                )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            optimization_errors.inc()
            raise

# Initialize optimizer
optimizer = ModelOptimizer()

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    await optimizer.initialize()

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_model(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Optimize a model"""
    try:
        result = await optimizer.optimize_model(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return generate_latest()

@app.get("/models")
async def list_models():
    """List optimized models"""
    models = []
    for model_file in optimizer.cache_dir.glob("*.trt"):
        models.append({
            "name": model_file.stem,
            "path": str(model_file),
            "size_mb": os.path.getsize(model_file) / (1024 * 1024),
            "modified": os.path.getmtime(model_file)
        })
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)