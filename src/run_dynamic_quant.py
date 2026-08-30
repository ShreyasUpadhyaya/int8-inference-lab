"""INT8 dynamic quantization: quantize_dynamic on Linear layers.

Dynamic quantization mechanics
-------------------------------
torch.quantization.quantize_dynamic swaps each targeted layer *type* for a
quantized equivalent whose weights are converted to int8 once, up front,
from the real float32 weights already sitting in the layer (affine
min/max quantization: scale = (max-min)/255, with the zero-point chosen so
real 0.0 maps to an exact integer). Weights don't change between calls, so
quantizing them ahead of time is free and safe.

Activations are handled differently, and this is the part that makes it
"dynamic": there is no calibration step. At every forward call, the
quantized layer computes the min/max of *that batch's own* input on the
fly, derives a scale/zero-point from it, quantizes the activation to int8,
runs an int8 x int8 matmul, then immediately dequantizes the result back
to float32 before handing it to the next (still-FP32) layer. That
per-call min/max + quantize/dequantize is real work -- dynamic
quantization only pays off when the matmul it wraps is expensive enough to
outweigh that bookkeeping. No qconfig, no observers, no prepare/convert
split -- those belong to static quantization (commit 6).

Why this is a weak case for THIS model
----------------------------------------
quantize_dynamic only touches the layer types you pass it -- here,
nn.Linear. ResNet-18 has exactly one: the final 512->10 classifier head.
Every conv and every BatchNorm -- where essentially all of the network's
FLOPs and parameters live -- stays FP32. So this commit quantizes 5,120
weights out of roughly 11.2M. Expect latency and model size to barely
move. That's not the harness malfunctioning; it's the actual finding:
dynamic quantization is trivial to apply but its payoff on a conv-heavy
vision network is small. It earns its keep on Linear/LSTM-heavy models
(transformers, seq2seq RNNs), not here.
"""

import torch
import torch.nn as nn

from src.benchmark import benchmark_model, save_results
from src.data import get_cifar10_loaders
from src.model import load_checkpoint
from src.utils import set_seed

CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"
RESULTS_PATH = "results/dynamic_int8.json"
FP32_CPU_REFERENCE_PATH = "results/fp32_cpu_reference.json"


def print_results(results: dict) -> None:
    print(f"config:          {results['config']}")
    print(f"device:          {results['device']}")
    print(f"median latency:  {results['median_latency_ms']:.3f} ms")
    print(f"p95 latency:     {results['p95_latency_ms']:.3f} ms")
    print(f"model size:      {results['model_size_mb']:.2f} MB")
    print(f"top-1 accuracy:  {results['top1_accuracy']:.2f}%")


def main():
    set_seed(42)
    # Dynamic-quantized int8 kernels run through fbgemm/qnnpack, a CPU-only
    # path -- there's no CUDA int8 GEMM being invoked here. So this always
    # benchmarks on CPU, regardless of what device the FP32 baseline used.
    # That means the dynamic-INT8 row and results/baseline.json's FP32 row
    # differ in hardware (CPU vs GPU) as well as precision. To keep that
    # from muddying the quantization comparison, we also benchmark the
    # *unquantized* model on CPU below, as a same-hardware reference point:
    # the honest "what did INT8 buy me" comparison is dynamic-INT8-on-CPU
    # vs FP32-on-CPU, not dynamic-INT8-on-CPU vs FP32-on-GPU.
    device = "cpu"

    _, test_loader = get_cifar10_loaders()
    fp32_model = load_checkpoint(CHECKPOINT_PATH, device=device)

    print("--- FP32 on CPU (same-hardware reference for the row below) ---")
    fp32_cpu_results = benchmark_model(
        fp32_model,
        test_loader,
        config_name="fp32_pytorch_cpu_reference",
        device=device,
    )
    print_results(fp32_cpu_results)
    save_results(fp32_cpu_results, FP32_CPU_REFERENCE_PATH)
    print(f"-> saved to {FP32_CPU_REFERENCE_PATH}")

    linear_layers = [n for n, m in fp32_model.named_modules() if isinstance(m, nn.Linear)]
    print(f"\nLayers targeted for dynamic quantization: {linear_layers}")

    quantized_model = torch.quantization.quantize_dynamic(
        fp32_model, {nn.Linear}, dtype=torch.qint8,
    )

    print("--- INT8 dynamic on CPU ---")
    results = benchmark_model(
        quantized_model,
        test_loader,
        config_name="int8_dynamic_pytorch",
        device=device,
    )
    print_results(results)
    save_results(results, RESULTS_PATH)
    print(f"-> saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
