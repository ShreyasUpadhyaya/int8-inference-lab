"""FP32 baseline: benchmark the unmodified fine-tuned model on GPU.

Run after src/finetune.py has produced checkpoints/resnet18_cifar10.pth,
in the same Colab session you'll run the INT8 commits in later -- latency
numbers from different sessions aren't comparable (different T4
allocation, different background load), so the whole FP32-vs-INT8 story
has to come from one sitting.

    python -m src.finetune       # once, produces the checkpoint
    python -m src.run_baseline   # this script

Writes results/baseline.json.
"""

from src.benchmark import benchmark_model, save_results
from src.data import get_cifar10_loaders
from src.model import load_checkpoint
from src.utils import get_device, set_seed

CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pth"
RESULTS_PATH = "results/baseline.json"


def main():
    set_seed(42)
    device = get_device()
    if device != "cuda":
        print("Warning: no GPU detected -- this is meant to be the FP32 "
              "*GPU* baseline that INT8-on-CPU gets compared against. "
              "Check Runtime > Change runtime type in Colab.")

    _, test_loader = get_cifar10_loaders()
    model = load_checkpoint(CHECKPOINT_PATH, device=device)

    results = benchmark_model(
        model,
        test_loader,
        config_name="fp32_pytorch",
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
