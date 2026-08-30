"""One-time fine-tuning of ResNet-18's new stem + head on CIFAR-10.

Run this once per Colab session, before any benchmarking commit:

    python -m src.finetune

It writes checkpoints/resnet18_cifar10.pth, which every later script
(benchmark harness, dynamic/static quantization, ONNX export) loads via
src.model.load_checkpoint(). The checkpoint isn't committed to git (see
.gitignore) -- it's a session artifact, cheap enough to regenerate
(a few minutes on a T4) whenever a session starts fresh, and committing a
~45MB binary that changes every retrain would just bloat the repo.

Why fine-tune at all instead of using the pretrained weights directly:
build_resnet18_cifar() (src/model.py) replaces the stem and the final fc
layer with freshly-initialized weights sized for CIFAR-10. Those layers
start out random, so the model needs to actually learn CIFAR-10 before its
predictions mean anything -- this is what makes the FP32 top-1 accuracy
below a real baseline instead of an artifact of a label-space mismatch.
"""

import os
import time

import torch.nn as nn
import torch.optim as optim

from src.data import get_cifar10_loaders
from src.evaluate import evaluate_accuracy
from src.model import build_resnet18_cifar, save_checkpoint
from src.utils import get_device, set_seed

CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"
EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    if device == "cpu":
        print("Warning: no GPU detected. Fine-tuning on CPU will be slow "
              "(check Runtime > Change runtime type in Colab).")

    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

    model = build_resnet18_cifar(num_classes=10, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate_accuracy(model, test_loader, device)
        elapsed = time.time() - start

        print(f"epoch {epoch:2d}/{EPOCHS}  loss {train_loss:.4f}  "
              f"test_acc {test_acc:.2f}%  ({elapsed:.1f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, CHECKPOINT_PATH)

    print(f"Best top-1 accuracy: {best_acc:.2f}% -> saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
