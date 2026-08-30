"""ONNX Runtime static quantization: calibration + QOperator format.

Static quantization mechanics (ONNX Runtime side)
----------------------------------------------------
Same idea as PyTorch's static path (commit 6): calibrate on real data to
fix activation scale/zero-points ahead of time, instead of recomputing
them on every call the way dynamic quantization does (commit 9). The API
shape differs though -- quantize_static owns the calibration loop
internally; you hand it a CalibrationDataReader (get_next() returns one
batch as a {input_name: numpy_array} dict, or None when exhausted)
instead of writing the loop yourself.

Two things this script does that commit 9 didn't need to:

1. quant_pre_process (shape inference + constant folding on the graph)
   before quantization. ORT's own tooling warns about skipping this --
   the commit 9 run logged exactly that warning. Static quantization's
   calibration and node-fusion decisions depend on knowing tensor shapes
   throughout the graph, so skipping this risks the quantizer missing
   fusable patterns or misjudging which nodes are safe to quantize.
2. quant_format=QuantFormat.QOperator, set explicitly rather than left at
   ORT's default (QDQ, which sandwiches original ops like Conv between
   QuantizeLinear/DequantizeLinear nodes -- more portable across
   execution providers, but not the fastest on the plain CPU EP this
   project benchmarks on). QOperator instead swaps each quantized op for
   its dedicated fused kernel directly (QLinearConv, QLinearAdd, ...) --
   the fused int8 kernel path commit 9 found ConvInteger was missing,
   which is what made dynamic quantization a regression there.
3. activation_type=QuantType.QUInt8, set explicitly alongside
   weight_type=QuantType.QInt8. This isn't optional on x64: ORT's fast
   quantized CPU kernels are built around u8 activations x s8 weights
   ("u8s8") using VNNI-style instructions. Leaving both at int8 ("s8s8")
   with QOperator format falls back to a slow reference kernel -- ORT's
   own quantizer warns about exactly this combination, and an earlier
   run of this script (before this fix) confirmed it empirically: s8s8
   landed *slower* than commit 9's dynamic-quant regression.

Calibration data comes from CIFAR-10's *train* split, same reasoning as
commit 6: calibrating on the test split would leak eval-set information
into how the model gets quantized.
"""

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.benchmark import save_results
from src.data import get_cifar10_loaders
from src.onnx_benchmark import benchmark_onnx_model
from src.utils import set_seed

FP32_ONNX_PATH = "checkpoints/resnet18_cifar10.onnx"
PREPROCESSED_ONNX_PATH = "checkpoints/resnet18_cifar10_preprocessed.onnx"
QUANTIZED_ONNX_PATH = "checkpoints/resnet18_cifar10_static_int8.onnx"
RESULTS_PATH = "results/onnx_static_int8.json"
NUM_CALIBRATION_BATCHES = 20  # same budget as commit 6's PyTorch calibration
INPUT_NAME = "input"  # matches input_names=["input"] from run_onnx_baseline.py's export


class CIFAR10CalibrationDataReader(CalibrationDataReader):
    """Feeds real CIFAR-10 train batches to quantize_static's calibrator.

    quantize_static calls get_next() repeatedly until it returns None --
    it owns the loop, unlike PyTorch's manual calibrate() in commit 6.
    No rewind() override: the default MinMax calibration method this
    script uses only needs a single pass over the calibration data.
    """

    def __init__(self, loader, input_name: str, num_batches: int):
        self.input_name = input_name
        self.num_batches = num_batches
        self._iterator = iter(loader)
        self._batches_yielded = 0

    def get_next(self):
        if self._batches_yielded >= self.num_batches:
            return None
        try:
            images, _ = next(self._iterator)
        except StopIteration:
            return None
        self._batches_yielded += 1
        return {self.input_name: images.numpy()}


def main():
    set_seed(42)

    print(f"Pre-processing (shape inference) -> {PREPROCESSED_ONNX_PATH}")
    quant_pre_process(FP32_ONNX_PATH, PREPROCESSED_ONNX_PATH)

    train_loader, test_loader = get_cifar10_loaders()
    calibration_reader = CIFAR10CalibrationDataReader(
        train_loader, INPUT_NAME, NUM_CALIBRATION_BATCHES,
    )

    print(f"Calibrating on {NUM_CALIBRATION_BATCHES} batches of CIFAR-10 train data...")
    quantize_static(
        model_input=PREPROCESSED_ONNX_PATH,
        model_output=QUANTIZED_ONNX_PATH,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    results = benchmark_onnx_model(
        QUANTIZED_ONNX_PATH,
        test_loader,
        config_name="int8_static_onnxruntime",
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
