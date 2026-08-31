"""Layer sensitivity analysis: which layers cost the most accuracy under
INT8 weight quantization, and why.

Method
------
For each Conv2d/Linear layer in turn: replace its weight tensor with a
per-channel, symmetric int8 fake-quantized version (quantize then
immediately dequantize -- the exact rounding/clipping error real int8
weight quantization would introduce, but the layer still runs as an
ordinary float32 op), evaluate top-1 accuracy on the full CIFAR-10 test
set, record the drop from the FP32 baseline, then restore the original
weight before moving to the next layer. Every other layer stays
untouched FP32 for that run -- so the accuracy drop measured is
attributable to exactly one layer's weight quantization error.

This deliberately measures *weight* quantization sensitivity only, not
activation quantization noise. That's a real scoping choice, not an
oversight: reproducing the full static-quantization graph (commit 6/10)
independently for all ~21 layers would mean inserting QuantStub/
DeQuantStub boundaries at 21 different points in the network, which is a
lot of fragile per-layer graph surgery for what this analysis needs. Per-
channel symmetric quantization matches the scheme our actual static
quantization runs use for weights (PerChannelMinMaxObserver,
qscheme=torch.per_channel_symmetric -- see commit 6), so this measures
the same kind of error our real quantized models actually have, just in
isolation, one layer at a time.

Hypothesis worth testing explicitly: a common finding in quantization
literature is that the *first* and *last* layers of a network are
disproportionately sensitive to quantization -- many practical
quantization recipes keep them at higher precision for exactly this
reason. Whether that holds for this specific CIFAR-adapted ResNet-18
(with its non-standard 3x3-stride-1 stem) is an empirical question this
script actually answers, rather than assumes.
"""

import torch
import torch.nn as nn

from src.benchmark import save_results
from src.data import get_cifar10_loaders
from src.evaluate import evaluate_accuracy
from src.model import load_checkpoint
from src.utils import get_device, set_seed

CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"
RESULTS_PATH = "results/layer_sensitivity.json"


def fake_quantize_weight_int8_per_channel(weight: torch.Tensor) -> torch.Tensor:
    """Simulate int8 quantize+dequantize on a weight tensor, per output
    channel (dim 0), symmetric (zero_point=0) -- matching the scheme
    PyTorch's default fbgemm qconfig uses for weights. Returns a float32
    tensor of the same shape holding the values int8 quantization would
    actually produce after dequantization.
    """
    reduce_dims = list(range(1, weight.dim()))  # everything except the output-channel dim
    max_abs = weight.detach().abs().amax(dim=reduce_dims)
    max_abs = max_abs.clamp(min=1e-8)  # avoid a divide-by-zero scale for an all-zero channel
    scale = max_abs / 127.0
    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    return torch.fake_quantize_per_channel_affine(
        weight, scale, zero_point, axis=0, quant_min=-128, quant_max=127,
    )


def get_quantizable_layers(model: nn.Module) -> list:
    """(name, module) for every Conv2d/Linear -- the layers that actually
    carry weights worth quantizing. BatchNorm is skipped deliberately:
    real static quantization folds it into the preceding conv (commit 6's
    fusion step) rather than quantizing it as an independent layer.
    """
    return [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def main():
    set_seed(42)
    device = get_device()  # fake-quantization is plain float math -- no CPU-only
                            # backend constraint, so GPU is fine and much faster
                            # for the ~21 full-test-set passes this needs.

    _, test_loader = get_cifar10_loaders()
    model = load_checkpoint(CHECKPOINT_PATH, device=device)

    baseline_accuracy = evaluate_accuracy(model, test_loader, device=device)
    print(f"FP32 baseline top-1: {baseline_accuracy:.2f}%")

    layers = get_quantizable_layers(model)
    print(f"Testing {len(layers)} layers one at a time...\n")

    per_layer_results = []
    for name, module in layers:
        original_weight = module.weight.data.clone()
        module.weight.data = fake_quantize_weight_int8_per_channel(original_weight)

        accuracy = evaluate_accuracy(model, test_loader, device=device)
        accuracy_drop_pp = baseline_accuracy - accuracy

        module.weight.data = original_weight  # restore before testing the next layer

        per_layer_results.append({
            "layer": name,
            "layer_type": type(module).__name__,
            "num_weights": original_weight.numel(),
            "top1_accuracy": accuracy,
            "accuracy_drop_pp": accuracy_drop_pp,
        })
        print(f"{name:28s} ({type(module).__name__:6s}, {original_weight.numel():>7,} wts)  "
              f"top1: {accuracy:6.2f}%  drop: {accuracy_drop_pp:+.2f}pp")

    per_layer_results.sort(key=lambda r: r["accuracy_drop_pp"], reverse=True)

    print("\nMost sensitive layers (largest accuracy drop when quantized alone):")
    for r in per_layer_results[:5]:
        print(f"  {r['layer']:28s} drop: {r['accuracy_drop_pp']:+.2f}pp")

    output = {
        "baseline_top1_accuracy": baseline_accuracy,
        "method": (
            "per-channel symmetric int8 fake-quantization of one layer's "
            "weights at a time (all other layers FP32); accuracy_drop_pp "
            "= baseline_top1_accuracy - top1_accuracy for that run"
        ),
        "results_sorted_by_sensitivity": per_layer_results,
    }
    save_results(output, RESULTS_PATH)
    print(f"\n-> saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
