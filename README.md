# int8-inference-lab

Benchmarking post-training INT8 quantization of ResNet-18 across two runtimes
-- PyTorch and ONNX Runtime -- measuring latency, model size, and top-1
accuracy for each configuration, on the same hardware, in the same session.

## Status

Work in progress. Results below are filled in as each configuration is
benchmarked (see `PHASE1_QUANTIZATION_PLAN.md` for the commit-by-commit plan).

## Results

| Config | Latency (ms) | Size (MB) | Top-1 | Δ Accuracy |
|---|---|---|---|---|
| FP32 (PyTorch) | | | | baseline |
| INT8 dynamic (PyTorch) | | | | |
| INT8 static (PyTorch) | | | | |
| FP32 (ONNX Runtime) | | | | |
| INT8 dynamic (ORT) | | | | |
| INT8 static (ORT) | | | | |

## Setup

Developed and run on Google Colab (free tier, T4 GPU). See
`PHASE1_QUANTIZATION_PLAN.md` for the Colab setup steps.

```bash
pip install -r requirements.txt
```

## Repo layout

- `src/` -- model loading, benchmarking harness, quantization scripts
- `results/` -- one JSON file per configuration, produced by the harness
