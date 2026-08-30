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
from torchvision.models.resnet import BasicBlock, ResNet


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


class QuantizableBasicBlock(BasicBlock):
    """BasicBlock with the residual add+relu made quantization-friendly.

    The stock BasicBlock reuses one `self.relu` module *object* twice:
    once after bn1, and again after the residual addition. `fuse_modules`
    fuses by name, not by tracing forward() -- fusing (conv1, bn1, relu)
    folds bn1's scale/shift into conv1 and replaces both bn1 and relu with
    nn.Identity in place. Since it's the same relu object called a second
    time after the residual add, that second, necessary ReLU silently
    becomes a no-op: a quantized model that's wrong, not approximate, and
    wrong in a way that raises no error.

    The fix (taken from torchvision's own quantization-ready ResNet) is to
    route the residual add through a FloatFunctional, which exposes
    add_relu() as its own distinct, observable op instead of reusing
    self.relu. FloatFunctional carries no learnable parameters, so a
    checkpoint trained on the plain BasicBlock (src/finetune.py's output)
    loads into this class with no key mismatch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.skip_add_relu.add_relu(out, identity)


def build_resnet18_cifar_quantizable(num_classes: int = 10) -> nn.Module:
    """Same architecture as build_resnet18_cifar, but built from
    QuantizableBasicBlock so static quantization's fuse_modules step
    (src/run_static_quant.py) doesn't silently drop the post-residual
    ReLU.

    No pretrained-weight loading here: this is only ever used via
    load_checkpoint_quantizable below, which immediately overwrites every
    weight with the already-fine-tuned checkpoint -- downloading ImageNet
    weights first would be wasted work.
    """
    model = ResNet(QuantizableBasicBlock, [2, 2, 2, 2], num_classes=1000)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_checkpoint_quantizable(path: str, num_classes: int = 10, device: str = "cpu") -> nn.Module:
    """Like load_checkpoint, but builds the fusion-friendly architecture
    that static quantization needs. QuantizableBasicBlock's extra
    FloatFunctional submodule carries no parameters, so this loads the
    exact same checkpoint file finetune.py produced -- if the keys ever
    didn't line up, load_state_dict would raise immediately rather than
    silently loading a mismatched model.
    """
    model = build_resnet18_cifar_quantizable(num_classes=num_classes)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
