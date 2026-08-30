"""CIFAR-10 dataset loading.

CIFAR-10 images are 32x32, far smaller than the 224x224 input torchvision's
pretrained ResNet-18 expects. We don't upscale CIFAR-10 to 224x224 -- that
would be wasteful and isn't how CIFAR-10 is normally benchmarked. Instead
src/model.py adapts the network's stem for small inputs, and this module
normalizes with CIFAR-10's own per-channel statistics rather than
ImageNet's, since the whole network gets fine-tuned end to end anyway (see
src/finetune.py).
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_cifar10_loaders(data_dir: str = "data", batch_size: int = 128, num_workers: int = 2):
    """Return (train_loader, test_loader) for CIFAR-10.

    Train transforms include light augmentation (random crop + horizontal
    flip) since we're fine-tuning a freshly-initialized stem and head. Test
    transforms are just tensor conversion + normalization -- no
    augmentation, since this is the fixed 10,000-image evaluation set every
    model configuration (FP32, INT8 dynamic, INT8 static, ONNX variants...)
    gets scored against. Keeping it fixed and un-augmented is what makes
    top-1 deltas between configs meaningful instead of noise.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
