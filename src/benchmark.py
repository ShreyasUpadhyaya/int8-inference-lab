"""Benchmarking harness: latency, model size, and accuracy in one place.

This is the part that makes every number in the README trustworthy or
worthless. Three things a naive benchmark gets wrong, all handled here:

1. A single time.time() call. CUDA kernel launches are asynchronous: the
   CPU issues the launch and moves on immediately, so one before/after
   timestamp measures "how long it took to queue the work", not "how long
   the GPU took to actually do it". Fixed with torch.cuda.synchronize()
   immediately before starting the clock and immediately after stopping
   it, so the timed interval brackets real GPU completion, not just
   kernel-launch overhead.
2. No warmup. The first few forward passes pay for CUDA context init,
   cuDNN algorithm autotuning, and allocator growth -- none of which
   reflect steady-state inference cost. Fixed with >=10 untimed warmup
   iterations before any measurement starts.
3. Mean over a handful of runs. Latency on shared, noisy hardware (Colab's
   T4 is shared infrastructure) has a long right tail -- one slow
   iteration from a scheduling hiccup drags the mean up and hides the
   number that describes normal-case performance. Fixed by running 100+
   timed iterations and reporting the median (typical case) and p95
   (tail behavior) instead of a mean over a handful of runs.
"""

import json
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluate import evaluate_accuracy


def get_model_size_mb(model: nn.Module) -> float:
    """Size of the model's state_dict on disk, in MB.

    Actually serializes to a temp file rather than hand-summing
    parameter.numel() * dtype.itemsize, because the hand-rolled version
    misses buffers (e.g. BatchNorm running stats) and -- the whole point
    of measuring this at all -- doesn't reflect how quantized modules
    actually store weights (packed int8 tensors plus scale/zero-point
    metadata, not "same layout, smaller dtype"). Writing the real
    state_dict and checking the file size measures what quantization
    actually changes on disk.
    """
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        os.remove(tmp_path)
    return size_bytes / (1024 ** 2)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape=(1, 3, 32, 32),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Median and p95 latency (ms) for one forward pass at `input_shape`.

    Batch size 1 by default: this measures single-request latency (how
    long one inference takes), which is the number that matters for
    "acceleration" claims, as opposed to throughput (images/sec at a large
    batch), which is a different question with a different harness.

    `device` is CPU by default because PyTorch's static/dynamic INT8
    quantized modules run on CPU backends (fbgemm/qnnpack) -- pass
    device="cuda" explicitly for the FP32 baseline. See CLAUDE.md: this
    harness deliberately does not silently move a quantized model to CUDA
    for you.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Warmup: untimed. Lets CUDA context init / cuDNN autotuning / allocator
    # growth happen before we start measuring steady-state cost.
    for _ in range(num_warmup):
        _ = model(dummy_input)
    if is_cuda:
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(num_runs):
        if is_cuda:
            torch.cuda.synchronize()  # drain any in-flight work before starting the clock
        start = time.perf_counter()

        _ = model(dummy_input)

        if is_cuda:
            torch.cuda.synchronize()  # block until this iteration's kernels actually finish
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    return {
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "num_warmup": num_warmup,
        "num_runs": num_runs,
        "all_latencies_ms": latencies_ms,
    }


def benchmark_model(
    model: nn.Module,
    test_loader: DataLoader,
    config_name: str,
    input_shape=(1, 3, 32, 32),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Full benchmark for one model configuration: latency + size + accuracy.

    Every commit from here on (FP32 baseline, dynamic INT8, static INT8,
    ONNX variants) calls this same function -- same accuracy procedure,
    same latency methodology, same size measurement -- so the only thing
    that differs between rows in the results table is the model and
    device, not the measurement approach.
    """
    latency = measure_latency(
        model, input_shape=input_shape, device=device,
        num_warmup=num_warmup, num_runs=num_runs,
    )
    size_mb = get_model_size_mb(model)
    top1_accuracy = evaluate_accuracy(model, test_loader, device=device)

    return {
        "config": config_name,
        "device": device,
        "median_latency_ms": latency["median_latency_ms"],
        "p95_latency_ms": latency["p95_latency_ms"],
        "num_warmup": latency["num_warmup"],
        "num_runs": latency["num_runs"],
        "model_size_mb": size_mb,
        "top1_accuracy": top1_accuracy,
    }


def save_results(results: dict, path: str) -> None:
    """Write a benchmark result dict to JSON immediately.

    Per CLAUDE.md: save to disk right after each run rather than holding
    results in memory across a Colab session that can die at any moment
    (idle timeout, disconnect, crash). Losing an in-memory dict costs you
    the whole benchmark run; losing this costs you nothing.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
