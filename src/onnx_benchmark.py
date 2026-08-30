"""Benchmarking helpers for ONNX Runtime models.

Mirrors src/benchmark.py's shape (same result dict keys, same median/p95/
warmup discipline) so PyTorch rows and ONNX Runtime rows in results/*.json
are directly comparable. Kept as a separate module rather than folded into
benchmark.py because the mechanics genuinely differ: an InferenceSession
takes numpy arrays through session.run(), not torch tensors through a
forward call, and CPUExecutionProvider's run() is synchronous -- there's
no async kernel-launch gap to bridge with torch.cuda.synchronize() the way
there is on GPU.
"""

import os
import time

import numpy as np
import onnxruntime as ort


def get_onnx_model_size_mb(onnx_path: str) -> float:
    return os.path.getsize(onnx_path) / (1024 ** 2)


def measure_latency_onnx(
    session: ort.InferenceSession,
    input_name: str,
    input_shape=(1, 3, 32, 32),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    latencies_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    return {
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "num_warmup": num_warmup,
        "num_runs": num_runs,
    }


def evaluate_accuracy_onnx(session: ort.InferenceSession, input_name: str, loader) -> float:
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = session.run(None, {input_name: images.numpy()})[0]
        predictions = outputs.argmax(axis=1)
        correct += int((predictions == labels.numpy()).sum())
        total += labels.size(0)
    return 100.0 * correct / total


def benchmark_onnx_model(
    onnx_path: str,
    test_loader,
    config_name: str,
    input_shape=(1, 3, 32, 32),
    num_warmup: int = 10,
    num_runs: int = 100,
    providers=("CPUExecutionProvider",),
) -> dict:
    """Full benchmark for one ONNX model: latency + size + accuracy.

    CPUExecutionProvider only, deliberately -- ONNX Runtime's dynamic and
    static quantization tooling (commits 9-10) targets CPU the same way
    PyTorch's does, so keeping the FP32-ONNX baseline on CPU too keeps
    every ONNX Runtime row in the final table on the same hardware.
    """
    session = ort.InferenceSession(onnx_path, providers=list(providers))
    input_name = session.get_inputs()[0].name

    latency = measure_latency_onnx(
        session, input_name, input_shape=input_shape,
        num_warmup=num_warmup, num_runs=num_runs,
    )
    size_mb = get_onnx_model_size_mb(onnx_path)
    top1_accuracy = evaluate_accuracy_onnx(session, input_name, test_loader)

    return {
        "config": config_name,
        "device": "cpu",
        "median_latency_ms": latency["median_latency_ms"],
        "p95_latency_ms": latency["p95_latency_ms"],
        "num_warmup": latency["num_warmup"],
        "num_runs": latency["num_runs"],
        "model_size_mb": size_mb,
        "top1_accuracy": top1_accuracy,
    }
