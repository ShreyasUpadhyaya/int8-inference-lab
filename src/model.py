"""ResNet-18 adapted for CIFAR-10.

torchvision's pretrained ResNet-18 is trained on ImageNet: 1000 classes,
224x224 inputs. CIFAR-10 has 10 classes and 32x32 inputs, so two things
have to change before the pretrained weights are usable at all:

1. The stem. ImageNet's stem is a 7x7 stride-2 conv followed by a 3x3
   stride-2 maxpool -- a 4x spatial downsample before the first residual
   block, tuned for 224x224 input. Run that on a 32x32 CIFAR image and
   you're down to 8x8 before any real feature extraction happens. The
   standard fix (from the original ResNet paper's CIFAR variant) is a 3x3
   stride-1 conv with no maxpool: keep full resolution going into the
   first residual stage.
2. The head. The final fc layer is a 512 -> 1000 classifier for ImageNet's
   label set. We replace it with 512 -> 10 for CIFAR-10 -- ImageNet and
   CIFAR-10 don't share a label space, so the old head is not just wrong
   sized, it's meaningless here regardless of size.

Both replacements are freshly initialized (there's no pretrained weight
for a 10-class output or a 3x3 stem conv), so the model needs fine-tuning
before it produces meaningful predictions -- see src/finetune.py. We keep
the pretrained weights for everything else (conv2_x through conv5_x,
i.e. all four residual stages): those layers encode general low/mid-level
visual features (edges, textures, shape parts) that transfer well and
give faster, better convergence than training from scratch.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18_cifar(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # CIFAR stem: 3x3 stride-1 conv, no maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 10-class head.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(path: str, num_classes: int = 10, device: str = "cpu") -> nn.Module:
    """Build the CIFAR-10 ResNet-18 architecture and load fine-tuned weights.

    Every benchmarking/quantization script (commit 3 onward) loads the
    model this way, not via build_resnet18_cifar directly -- they need the
    fine-tuned weights, not a fresh ImageNet-initialized stem+backbone.

    pretrained=False here: we're about to overwrite every weight with the
    fine-tuned checkpoint, so there's no reason to spend time downloading
    ImageNet weights first.
    """
    model = build_resnet18_cifar(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
