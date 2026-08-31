"""Generates results/latency_comparison.png from the results/*.json files.

Pure local plotting, no GPU/session needed -- run any time after the
benchmark JSON files exist:

    python scripts/plot_results.py
"""

import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "latency_comparison.png")

BLUE = "#2a78d6"    # categorical slot 1 -- PyTorch
ORANGE = "#eb6834"  # categorical slot 2 -- ONNX Runtime
GRID_GRAY = "#d9d8d3"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def load(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)


def main():
    fp32_gpu = load("baseline")
    fp32_cpu = load("fp32_cpu_reference")
    pt_dynamic = load("dynamic_int8")
    pt_static = load("static_int8")
    onnx_fp32 = load("onnx_baseline")
    onnx_dynamic = load("onnx_dynamic_int8")
    onnx_static = load("onnx_static_int8")

    categories = ["FP32 (CPU)", "INT8 dynamic", "INT8 static"]
    pytorch_values = [
        fp32_cpu["median_latency_ms"],
        pt_dynamic["median_latency_ms"],
        pt_static["median_latency_ms"],
    ]
    onnx_values = [
        onnx_fp32["median_latency_ms"],
        onnx_dynamic["median_latency_ms"],
        onnx_static["median_latency_ms"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    x = range(len(categories))
    width = 0.32
    bar_kwargs = dict(width=width, edgecolor="#fcfcfb", linewidth=2)

    bars1 = ax.bar(
        [i - width / 2 - 0.01 for i in x], pytorch_values,
        color=BLUE, label="PyTorch", **bar_kwargs,
    )
    bars2 = ax.bar(
        [i + width / 2 + 0.01 for i in x], onnx_values,
        color=ORANGE, label="ONNX Runtime", **bar_kwargs,
    )

    # Direct labels at each bar's tip -- the story is the actual ms values.
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY,
            )

    # FP32-GPU reference line -- different hardware, shown for context only,
    # not part of the CPU-vs-CPU comparison the bars make.
    gpu_latency = fp32_gpu["median_latency_ms"]
    ax.axhline(gpu_latency, color=TEXT_SECONDARY, linestyle="--", linewidth=1.5, zorder=0)
    ax.annotate(
        f"FP32 on GPU: {gpu_latency:.2f}ms (different hardware, context only)",
        xy=(len(categories) - 1, gpu_latency),
        xytext=(0, 6), textcoords="offset points",
        ha="right", va="bottom", fontsize=8.5, color=TEXT_SECONDARY, style="italic",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Median latency, batch size 1 (ms)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(
        "ResNet-18 / CIFAR-10: inference latency by config",
        fontsize=13, color=TEXT_PRIMARY, loc="left", pad=14,
    )

    ax.grid(axis="y", color=GRID_GRAY, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_GRAY)
    ax.tick_params(left=False, bottom=False, colors=TEXT_SECONDARY)

    ax.legend(frameon=False, loc="upper left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, facecolor=fig.get_facecolor())
    print(f"-> saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
