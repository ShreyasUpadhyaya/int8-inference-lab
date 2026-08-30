# int8-inference-lab

Benchmarking post-training INT8 quantization of ResNet-18 across two runtimes
-- PyTorch and ONNX Runtime -- measuring latency, model size, and top-1
accuracy for each configuration, on the same hardware, in the same session.

## Status

PyTorch path (FP32, INT8 dynamic, INT8 static) and the ONNX Runtime FP32
baseline are done. ONNX Runtime INT8 (dynamic/static) and the layer
sensitivity analysis are in progress -- see `PHASE1_QUANTIZATION_PLAN.md`
for the commit-by-commit plan.

## Results

Every row below is the same fine-tuned ResNet-18 checkpoint (94.14% top-1
in isolation), scored on the same fixed 10,000-image CIFAR-10 test set,
in one Kaggle session (T4 GPU + its host CPU) -- checkpoint, hardware,
and session all held constant so latency and accuracy deltas are actually
attributable to the precision/runtime change in each row, not to
re-fine-tuning variance or a different machine. Latency is single-image
(batch size 1), median and p95 over 100 timed runs after 10 warmup
iterations.

| Config | Runtime | Device | Median (ms) | p95 (ms) | Size (MB) | Top-1 | Δ Accuracy |
|---|---|---|---|---|---|---|---|
| FP32 | PyTorch | GPU | 2.820 | 4.212 | 42.70 | 94.14% | *(different hardware -- context only)* |
| FP32 | PyTorch | CPU | 12.275 | 13.241 | 42.70 | 94.14% | baseline |
| INT8 dynamic | PyTorch | CPU | 11.915 | 13.341 | 42.69 | 94.14% | 0.00 pp |
| INT8 static | PyTorch | CPU | 6.598 | 7.269 | 10.78 | 94.14% | 0.00 pp |
| FP32 | ONNX Runtime | CPU | 11.674 | 12.549 | 42.62 | 94.14% | 0.00 pp |
| INT8 dynamic | ONNX Runtime | CPU | TBD | TBD | TBD | TBD | TBD |
| INT8 static | ONNX Runtime | CPU | TBD | TBD | TBD | TBD | TBD |

**Why there are two FP32-PyTorch rows:** PyTorch's eager-mode INT8
kernels (dynamic and static alike) run through CPU backends
(fbgemm/qnnpack) -- there is no CUDA path for them. Comparing INT8-on-CPU
latency against FP32-on-GPU would conflate "the GPU is faster than a CPU
core" with "quantization made this faster," a different claim. The
FP32-CPU row is the fair, same-hardware baseline the INT8 rows should
actually be read against; the FP32-GPU row is there because it's the
number a reader instinctively expects to see, not because it's a valid
comparator for the rows below it.

**Dynamic quantization** (`torch.quantization.quantize_dynamic`) converts
weights to int8 once, ahead of time, but only for the layer types you
target -- here, `nn.Linear`. ResNet-18 has exactly one Linear layer: the
final 512-unit classifier head, about 5,120 weights out of the network's
~11.2M. Every convolution and BatchNorm -- where nearly all the compute
and parameters actually live -- stays FP32. Activations for that one
Linear layer get quantized on the fly at every call (their min/max
computed per-batch, with no calibration step), which is itself a small
amount of overhead. The result above is exactly what that combination
predicts: no change in size, no change in accuracy, and latency that's a
wash -- within noise of FP32-CPU either way. Dynamic quantization's win
depends on the quantized op being expensive enough to outweigh its own
bookkeeping, and a 512x10 matmul isn't.

**Static quantization** quantizes the convolutions too, which is where
the payoff actually is on a conv-heavy network. It costs a calibration
pass (20 batches of real CIFAR-10 training data shown to the model,
purely to let observers record realistic activation ranges -- no
gradient, no weight update) and a module-fusion step (folding each
BatchNorm into its preceding conv so the observer sees one quantized op's
real output instead of a float BN sitting between two int8 boundaries).
The result: **~1.86x faster than FP32-CPU** (6.598ms vs 12.275ms),
**~3.96x smaller on disk** (10.78MB vs 42.70MB, close to the theoretical
4x ceiling for fp32->int8), with zero measured accuracy cost on this
test set. Static quantization needs more setup than dynamic (a qconfig,
a calibration dataset, fuse/prepare/convert instead of one function
call), but on this architecture it earns that setup back with a real
speedup that dynamic quantization simply doesn't deliver.

**ONNX Runtime's FP32 baseline** is ~4.9% faster than PyTorch's FP32-CPU
(11.674ms vs 12.275ms) at identical accuracy -- the same weights, the
same precision, just a different execution engine. That gap is from
ONNX Runtime's graph-level optimizations (constant folding, operator
fusion) rather than anything quantization-related, which is exactly why
this baseline was worth measuring on its own before adding ONNX Runtime's
INT8 paths (commits 9-10): without it, any ORT-INT8 speedup claim would
be ambiguous about which effect -- runtime or precision -- it's actually
crediting.

## Setup

Developed and run on Google Colab and Kaggle Notebooks (free tier, T4
GPU). See `PHASE1_QUANTIZATION_PLAN.md` for the Colab setup steps.

```bash
pip install -r requirements.txt
```

## Repo layout

- `src/` -- model loading, benchmarking harness, quantization scripts
- `results/` -- one JSON file per configuration, produced by the harness
