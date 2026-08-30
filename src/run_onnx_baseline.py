"""ONNX export + FP32 baseline through ONNX Runtime.

Why benchmark ONNX Runtime's FP32 model at all, before any quantization:
swapping PyTorch's eager-mode execution for ONNX Runtime's graph-level
execution (constant folding, operator fusion, a different set of
hand-optimized CPU kernels) can shift latency on its own, independent of
precision. Without this FP32-on-ORT number, a later "ORT INT8 is Nx
faster" claim would be ambiguous -- faster than ORT's own FP32, or faster
than PyTorch's FP32 baseline? Those are different, conflatable claims,
the same category of mistake as the GPU-vs-CPU mixup flagged back in
commit 5.

torch.onnx.export mechanics
-----------------------------
torch.onnx.export traces one real forward pass with the dummy input,
records every op that ran, and serializes that as a static computation
graph (the ONNX format) -- not a copy of the Python code. Two
consequences worth knowing:
- Any data-dependent control flow (an `if` on a tensor value, a Python
  loop with a tensor-dependent trip count) gets "baked in" to whichever
  branch happened to execute during tracing. Not a concern for
  ResNet-18 -- no such branching -- but the reason tracing-based export
  isn't a universal drop-in for every model.
- `dynamic_axes` marks the batch dimension as variable instead of frozen
  at whatever batch size the dummy input used. Without it, this exported
  graph would only accept batch size 1 (the latency harness's shape) and
  would reject the batch-128 tensors evaluate_accuracy_onnx feeds it from
  the CIFAR-10 test loader.

This always benchmarks on CPUExecutionProvider -- see
src/onnx_benchmark.py's docstring for why.
"""

import onnx
import torch

from src.benchmark import save_results
from src.data import get_cifar10_loaders
from src.model import load_checkpoint
from src.onnx_benchmark import benchmark_onnx_model
from src.utils import set_seed

CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"
ONNX_PATH = "checkpoints/resnet18_cifar10.onnx"
RESULTS_PATH = "results/onnx_baseline.json"


def export_to_onnx(model: torch.nn.Module, onnx_path: str) -> None:
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
        # PyTorch >=2.5 defaults torch.onnx.export to a newer TorchDynamo-
        # based exporter that requires the separate `onnxscript` package.
        # dynamo=False pins to the older TorchScript-tracing exporter
        # (the mechanics this module's docstring describes) so this runs
        # with only what's already in requirements.txt.
        dynamo=False,
    )

    # Sanity-check the exported graph is well-formed before trusting any
    # benchmark run on it -- catches a malformed export loudly instead of
    # producing silently wrong latency/accuracy numbers downstream.
    onnx.checker.check_model(onnx.load(onnx_path))


def main():
    set_seed(42)
    device = "cpu"

    _, test_loader = get_cifar10_loaders()
    model = load_checkpoint(CHECKPOINT_PATH, device=device)

    print(f"Exporting to {ONNX_PATH}...")
    export_to_onnx(model, ONNX_PATH)
    print("Export OK (onnx.checker passed).")

    results = benchmark_onnx_model(
        ONNX_PATH,
        test_loader,
        config_name="fp32_onnxruntime",
    )

    print(f"config:          {results['config']}")
    print(f"device:          {results['device']}")
    print(f"median latency:  {results['median_latency_ms']:.3f} ms")
    print(f"p95 latency:     {results['p95_latency_ms']:.3f} ms")
    print(f"model size:      {results['model_size_mb']:.2f} MB")
    print(f"top-1 accuracy:  {results['top1_accuracy']:.2f}%")

    save_results(results, RESULTS_PATH)
    print(f"-> saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
