# Phase 1 — Model Quantization & Benchmarking

**Goal:** A repo that measures FP32 vs INT8 across PyTorch and ONNX Runtime, with real numbers for latency, model size, and accuracy.

**Time:** One weekend (2 days)
**Repo name:** `model-quantization-benchmark`
**Why this matters:** "Apply quantization and model compression (FP32 → lower precision)" is a listed responsibility on the AMD Model Optimization role, and appears in required skills. Right now you have zero evidence for it.

---

## Setup: Google Colab

You don't need a local GPU. Colab's free tier gives you a T4.

1. Go to https://colab.research.google.com
2. New notebook → Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save
3. Verify with:
   ```python
   !nvidia-smi
   ```
4. Mount Drive so work survives disconnects:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Connect the repo (generate a GitHub personal access token first, with `repo` scope):
   ```python
   !git config --global user.email "shreyasups@gmail.com"
   !git config --global user.name "Shreyas Upadhyaya"
   !git clone https://<TOKEN>@github.com/ShreyasUpadhyaya/model-quantization-benchmark.git
   %cd model-quantization-benchmark
   ```

**Colab caveats worth knowing:** sessions die after ~90 minutes idle and ~12 hours max. Commit often. Latency numbers vary between sessions, so run FP32 and INT8 benchmarks *in the same session* or the comparison is meaningless.

---

## Day 1 — Baseline and PyTorch quantization

### Commit 1: `chore: project scaffold`
- `requirements.txt`, `.gitignore`, empty `README.md`, `src/` and `results/` directories

### Commit 2: `feat: model loading and eval dataset`
- Load pretrained ResNet-18 from torchvision
- Load CIFAR-10 test set (or ImageNet-val subset), 500–1000 images
- Function that returns top-1 accuracy

### Commit 3: `feat: benchmarking harness`
This is the part that matters most. A sloppy harness makes every later number worthless.
- Warmup iterations before timing (at least 10)
- Median over 100+ runs, not mean, not a single run
- `torch.cuda.synchronize()` before and after timing on GPU
- Record: median latency, p95 latency, model size on disk, top-1 accuracy

### Commit 4: `feat: FP32 baseline results`
- Run the harness on the unmodified model
- Save to `results/baseline.json`

### Commit 5: `feat: dynamic quantization`
- `torch.quantization.quantize_dynamic` on Linear layers
- Re-run harness, save results

### Commit 6: `feat: static quantization with calibration`
- Set qconfig, prepare, run calibration batches, convert
- Re-run harness, save results

### Commit 7: `docs: day 1 comparison table`
- README table: FP32 vs dynamic INT8 vs static INT8

---

## Day 2 — ONNX Runtime path and analysis

### Commit 8: `feat: ONNX export and baseline`
- `torch.onnx.export` the FP32 model
- Run through ONNX Runtime, benchmark, save results

### Commit 9: `feat: ONNX Runtime dynamic quantization`
- `onnxruntime.quantization.quantize_dynamic`
- Benchmark, save

### Commit 10: `feat: ONNX Runtime static quantization`
- `quantize_static` with a calibration data reader
- Benchmark, save

### Commit 11: `feat: layer sensitivity analysis`
- Quantize layers one group at a time
- Record which layers cost the most accuracy
- This is the insight that separates you from someone who ran a tutorial

### Commit 12: `docs: full results and writeup`
- Complete comparison table, all 6 configurations
- A chart if time allows
- Short section: what each method does, where accuracy went, which layers were sensitive, when you'd choose each

---

## What "done" looks like

A README with a table like:

| Config | Latency (ms) | Size (MB) | Top-1 | Δ Accuracy |
|---|---|---|---|---|
| FP32 (PyTorch) | | | | baseline |
| INT8 dynamic (PyTorch) | | | | |
| INT8 static (PyTorch) | | | | |
| FP32 (ONNX Runtime) | | | | |
| INT8 dynamic (ORT) | | | | |
| INT8 static (ORT) | | | | |

Plus 3–4 paragraphs explaining the results.

---

## Resume line this earns you

`Model quantization (PyTorch, ONNX Runtime) | INT8 post-training quantization | Layer sensitivity analysis | Inference benchmarking`

## Interview answer this earns you

"I quantized ResNet-18 through both the PyTorch and ONNX Runtime paths. Static INT8 gave roughly Nx latency improvement and Y% size reduction for Z% top-1 loss. The accuracy cost was concentrated in [specific layers], which makes sense because [reason]. Dynamic quantization was easier to apply but only helped the Linear layers, so the win was smaller on a conv-heavy network."
