# int8-inference-lab

Benchmarking post-training INT8 quantization of ResNet-18 across two runtimes
-- PyTorch and ONNX Runtime -- measuring latency, model size, and top-1
accuracy for each configuration, on the same hardware, in the same session.

## Status

Day 1 (PyTorch: FP32 baseline, INT8 dynamic, INT8 static) is done. Day 2
(ONNX Runtime path + layer sensitivity analysis) is in progress -- see
`PHASE1_QUANTIZATION_PLAN.md` for the commit-by-commit plan.

## Results -- Day 1 (PyTorch)

All four rows are the same fine-tuned ResNet-18 checkpoint, scored on the
same fixed 10,000-image CIFAR-10 test set, in the same Colab session
(T4 GPU + its host CPU). Latency is single-image (batch size 1), median
and p95 over 100 timed runs after 10 warmup iterations.

| Config | Device | Median (ms) | p95 (ms) | Size (MB) | Top-1 | Δ Accuracy |
|---|---|---|---|---|---|---|
| FP32 (PyTorch) | GPU | 2.811 | 3.870 | 42.70 | 94.45% | *(different hardware -- context only, see note below)* |
| FP32 (PyTorch) | CPU | 17.416 | 21.784 | 42.70 | 93.96% | baseline |
| INT8 dynamic (PyTorch) | CPU | 18.026 | 22.542 | 42.69 | 93.95% | -0.01 pp |
| INT8 static (PyTorch) | CPU | 9.986 | 12.096 | 10.78 | 94.08% | +0.12 pp |

**Why there are two FP32 rows:** PyTorch's eager-mode INT8 kernels
(dynamic and static alike) run through CPU backends (fbgemm/qnnpack) --
there is no CUDA path for them. Comparing INT8-on-CPU latency against
FP32-on-GPU would conflate "the GPU is faster than a shared CPU core"
with "quantization made this faster," which is a different claim. The
FP32-CPU row is the fair, same-hardware baseline the two INT8 rows
should actually be read against; the FP32-GPU row is there because it's
the number a reader instinctively expects to see, not because it's a
valid comparator for the rows below it.

**Dynamic quantization** (`torch.quantization.quantize_dynamic`) converts
weights to int8 once, ahead of time, but only for the layer types you
target -- here, `nn.Linear`. ResNet-18 has exactly one Linear layer: the
final 512-unit classifier head, about 5,120 weights out of the network's
~11.2M. Every convolution and BatchNorm -- where nearly all the compute
and parameters actually live -- stays FP32. Activations for that one
Linear layer get quantized on the fly at every call (their min/max is
computed per-batch, with no calibration step), which is itself a small
amount of overhead. The result above shows exactly what that combination
predicts: essentially no change in size, no change in accuracy, and
latency that's a wash (marginally worse, if anything) rather than an
improvement -- dynamic quantization's win depends on the quantized op
being expensive enough to outweigh its own bookkeeping, and a 512x10
matmul isn't.

**Static quantization** quantizes the convolutions too, which is where
the payoff actually is on a conv-heavy network. It costs a calibration
pass (20 batches of real CIFAR-10 training data shown to the model,
purely to let observers record realistic activation ranges -- no
gradient, no weight update) and a module-fusion step (folding each
BatchNorm into its preceding conv so the observer sees one quantized op's
real output instead of a float BN sitting between two int8 boundaries).
The result: **~1.74x faster than FP32-CPU** (9.986ms vs 17.416ms),
**~3.96x smaller on disk** (10.78MB vs 42.70MB, close to the theoretical
4x ceiling for fp32->int8), and accuracy that didn't measurably drop --
94.08% vs 93.96% FP32, a difference well inside normal run-to-run noise.
Static quantization needs more setup than dynamic (a qconfig, a
calibration dataset, fuse/prepare/convert instead of one function call),
but on this architecture it earns that setup back with a real speedup
that dynamic quantization simply doesn't deliver.

## Results -- Day 2 (ONNX Runtime)

Pending -- ONNX export, ONNX Runtime dynamic/static quantization, and a
layer sensitivity analysis (commits 8-11).

## Setup

Developed and run on Google Colab (free tier, T4 GPU). See
`PHASE1_QUANTIZATION_PLAN.md` for the Colab setup steps.

```bash
pip install -r requirements.txt
```

## Repo layout

- `src/` -- model loading, benchmarking harness, quantization scripts
- `results/` -- one JSON file per configuration, produced by the harness
