"""INT8 static quantization: qconfig, calibration, prepare/convert.

Static quantization mechanics
------------------------------
Dynamic quantization (commit 5) only quantizes weights ahead of time and
recomputes activation ranges on every single call. Static quantization
commits to a fixed activation scale/zero-point *ahead of time* for every
quantized op in the network -- including the convolutions, which is where
this actually pays off on a convnet like ResNet-18. The cost is a
calibration step: you have to run real, representative data through the
model once before freezing those ranges.

The pipeline, in the order this script runs it:

1. Fuse Conv+BN(+ReLU) chains (`fuse_modules`). BatchNorm's scale/shift
   folds analytically into the preceding conv's weight and bias -- same
   math, fewer ops at inference, and it means the observer in the next
   step sees the real post-BN, post-ReLU activation distribution as one
   quantized op's output, instead of a quantized conv's output getting
   dequantized back to float just to run an unfused float BatchNorm.
   Skipping fusion doesn't break correctness, but it measurably costs
   both accuracy and speed. (See QuantizableBasicBlock in src/model.py
   for why the *second* relu per block is deliberately left out of this.)
2. Attach a qconfig: `get_default_qconfig("fbgemm")`. fbgemm is the x86
   CPU quantized-kernel backend (qnnpack is the ARM/mobile equivalent;
   Colab's CPU is x86). A qconfig is just a pair of *observer* types --
   one for weights, one for activations -- not a quantized value itself.
3. `prepare()` walks the fused model and inserts those observer modules
   after every quantizable op. Observers don't quantize anything yet --
   they watch float32 values pass through during calibration and record
   statistics (here: running min/max, per-tensor for activations,
   per-output-channel for weights).
4. Calibration: run several batches of real data through the prepared
   model in eval mode with gradients off. This is not training -- no
   loss, no backward pass, no weight update. Its only purpose is letting
   the observers see realistic activation ranges before those ranges get
   frozen. Calibration data comes from the CIFAR-10 *train* split here,
   never the test split the final accuracy number is reported on --
   calibrating on test data would leak eval-set information into how the
   model gets quantized.
5. `convert()` reads each observer's final min/max, computes a scale and
   zero-point from it (scale = (max - min) / 255 for int8's range, with
   the zero-point chosen so real 0.0 maps to an exact integer -- this
   matters because of zero-padding in convolutions), quantizes every
   weight tensor to int8 using those parameters, and swaps each fused
   float module for its quantized equivalent. After this call, tensors
   flow through the network as int8 end to end between the initial
   QuantStub and final DeQuantStub -- unlike dynamic quantization,
   nothing gets dequantized and requantized between layers.
"""

import torch
import torch.nn as nn

from src.benchmark import benchmark_model, save_results
from src.data import get_cifar10_loaders
from src.model import QuantizableBasicBlock, load_checkpoint_quantizable
from src.utils import set_seed

CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"
RESULTS_PATH = "results/static_int8.json"
NUM_CALIBRATION_BATCHES = 20  # ~2,560 CIFAR-10 train images at batch_size=128


def build_fusion_list(model: nn.Module) -> list:
    """Module-name triples to hand to fuse_modules.

    Built by walking the QuantWrapper-wrapped model's named modules
    rather than hand-writing "layer2.0.conv1"-style strings, so it stays
    correct regardless of exactly how many blocks each layer has or
    which blocks carry a downsample.
    """
    fusion_list = [["module.conv1", "module.bn1", "module.relu"]]
    for name, module in model.named_modules():
        if isinstance(module, QuantizableBasicBlock):
            fusion_list.append([f"{name}.conv1", f"{name}.bn1", f"{name}.relu"])
            fusion_list.append([f"{name}.conv2", f"{name}.bn2"])
            if module.downsample is not None:
                fusion_list.append([f"{name}.downsample.0", f"{name}.downsample.1"])
    return fusion_list


def calibrate(model: nn.Module, loader, num_batches: int) -> None:
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images)


def main():
    set_seed(42)
    device = "cpu"  # static quantization's fbgemm kernels are CPU-only
    torch.backends.quantized.engine = "fbgemm"  # x86 CPU backend

    train_loader, test_loader = get_cifar10_loaders()
    fp32_model = load_checkpoint_quantizable(CHECKPOINT_PATH, device=device)

    # QuantWrapper adds a QuantStub before the model and a DeQuantStub
    # after it, so the network operates on int8 tensors internally and
    # only converts to/from float at its actual input/output boundary.
    model = torch.quantization.QuantWrapper(fp32_model)
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    print(f"qconfig: {model.qconfig}")

    fusion_list = build_fusion_list(model)
    print(f"Fusing {len(fusion_list)} conv/bn(/relu) groups")
    torch.quantization.fuse_modules(model, fusion_list, inplace=True)

    torch.quantization.prepare(model, inplace=True)

    print(f"Calibrating on {NUM_CALIBRATION_BATCHES} batches of CIFAR-10 train data...")
    calibrate(model, train_loader, NUM_CALIBRATION_BATCHES)

    torch.quantization.convert(model, inplace=True)

    results = benchmark_model(
        model,
        test_loader,
        config_name="int8_static_pytorch",
        device=device,
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
