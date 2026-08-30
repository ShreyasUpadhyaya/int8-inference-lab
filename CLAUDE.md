# CLAUDE.md — int8-inference-lab

## What this project is

A benchmarking study of post-training INT8 quantization across two runtimes: PyTorch and ONNX Runtime. The deliverable is not "quantization works" — it's a reproducible measurement of what it costs and what it buys, on the same hardware, in the same session.

I'm building this to have real evidence for model optimization work, targeting AI Engineer roles focused on inference acceleration.

## Who I am

Shreyas Upadhyaya. 3 years professional experience, Python and C++. Strong on backend systems, ML application development, and profiling. New to quantization specifically. Don't explain Python basics or what a tensor is. Do explain quantization mechanics — I want to understand calibration, observers, qconfig, and why certain layers are sensitive.

## Environment

Google Colab, free tier, T4 GPU. Sessions die after ~90 min idle and ~12h max.

Implications you should respect:
- Any script must be runnable end to end from a fresh session
- Never assume prior state in memory
- FP32 and INT8 benchmarks must run in the SAME session — cross-session latency comparison is meaningless due to hardware variance
- Save results to JSON immediately after each run, don't hold them in memory

## Non-negotiables for the benchmarking harness

This is the part that makes or breaks the project. A sloppy harness makes every number worthless.

- Minimum 10 warmup iterations before any timing
- Median over 100+ runs, never mean, never a single `time.time()`
- `torch.cuda.synchronize()` before and after GPU timing — CUDA is async, timing without sync measures launch time not execution
- Report median AND p95 latency
- Fixed random seeds
- Record: latency, model size on disk, top-1 accuracy, for every config

If I ask for a shortcut that compromises measurement integrity, push back.

## Working style

- **One commit at a time.** Complete the work for a single commit, then stop. I review and commit myself.
- Use the commit messages from PHASE1_QUANTIZATION_PLAN.md.
- Explain the mechanics as you go. If you use `qconfig`, tell me what it's configuring and why that choice.
- Prefer clarity over cleverness. This repo is read by interviewers.
- No unnecessary abstraction. Small scripts beat a framework.

## What I care about in the output

The comparison table is the product. Everything else supports it.

The layer sensitivity analysis (commit 11) is the differentiator — anyone can run `quantize_dynamic`, few can say which layers cost the most accuracy and why. Don't rush that one.

## Stack

- PyTorch + torchvision (ResNet-18 pretrained)
- `torch.quantization` — dynamic and static
- ONNX + ONNX Runtime, `onnxruntime.quantization`
- CIFAR-10 test set, 500–1000 images for eval

## Things to avoid

- Don't fabricate or estimate benchmark numbers. If something didn't run, say so.
- Don't add MLflow, W&B, or experiment tracking. JSON files are enough.
- Don't write a config system. Hardcode and keep it readable.
- Don't quantize on GPU without checking backend support first — PyTorch static quantization is CPU-oriented (fbgemm/qnnpack). Flag this explicitly when we hit it rather than producing silently wrong results.

## Definition of done

README contains a 6-row comparison table with real numbers, plus 3–4 paragraphs explaining what each method does, where accuracy was lost, which layers were sensitive, and when I'd choose each approach.
