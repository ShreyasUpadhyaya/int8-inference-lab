"""Top-1 accuracy evaluation.

Deliberately separate from the benchmarking harness (src/benchmark.py):
this answers "how correct is the model", the harness answers "how fast is
the model". Every configuration -- FP32, INT8 dynamic, INT8 static, and
the ONNX Runtime equivalents -- gets scored with this same function
against the same fixed CIFAR-10 test set, so accuracy deltas between
configs reflect the quantization, not a different eval procedure.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Top-1 accuracy (%) of `model` over every batch in `loader`."""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total
