"""ONNX Runtime dynamic quantization.

onnxruntime.quantization.quantize_dynamic mirrors the mechanics of
torch.quantization.quantize_dynamic (commit 5): weights are converted to
int8 once, ahead of time; activations get their min/max computed on the
fly at every inference call, with no calibration step. The same
trade-off applies -- it only helps the specific ops it targets.

Which ops it targets isn't something worth asserting from memory: ORT's
default op_types_to_quantize list has shifted across versions, and
whether PyTorch's fc layer exported to a `Gemm` node or a `MatMul` node
(an export-mechanics detail, not something this script controls) decides
whether dynamic quantization touches it at all. Rather than assume, this
script counts operator types in the ONNX graph before and after
quantization and prints the diff -- results/onnx_dynamic_int8.json
records what actually happened to the graph, not what convention would
predict.
"""

import collections

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.benchmark import save_results
from src.data import get_cifar10_loaders
from src.onnx_benchmark import benchmark_onnx_model
from src.utils import set_seed

FP32_ONNX_PATH = "checkpoints/resnet18_cifar10.onnx"
QUANTIZED_ONNX_PATH = "checkpoints/resnet18_cifar10_dynamic_int8.onnx"
RESULTS_PATH = "results/onnx_dynamic_int8.json"


def count_op_types(onnx_path: str) -> collections.Counter:
    model = onnx.load(onnx_path)
    return collections.Counter(node.op_type for node in model.graph.node)


def main():
    set_seed(42)

    print(f"Op types before quantization: {dict(count_op_types(FP32_ONNX_PATH))}")

    quantize_dynamic(
        model_input=FP32_ONNX_PATH,
        model_output=QUANTIZED_ONNX_PATH,
        weight_type=QuantType.QInt8,
    )

    print(f"Op types after quantization:  {dict(count_op_types(QUANTIZED_ONNX_PATH))}")

    _, test_loader = get_cifar10_loaders()
    results = benchmark_onnx_model(
        QUANTIZED_ONNX_PATH,
        test_loader,
        config_name="int8_dynamic_onnxruntime",
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
