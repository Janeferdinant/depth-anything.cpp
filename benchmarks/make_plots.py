#!/usr/bin/env python3
"""Render the release benchmark plots for depth-anything.cpp.

Reads benchmarks/results.json (clean numbers) and writes PNGs under
benchmarks/media/. Pure-offline: matplotlib + numpy only, no network.

Run:  .venv/bin/python benchmarks/make_plots.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
os.makedirs(MEDIA, exist_ok=True)

# --- palette: PyTorch = warm orange, C++/ggml variants = one blue family -----
C_TORCH = "#E07B39"          # PyTorch (reference baseline)
C_CPP = {
    "f32": "#1F4E79",        # C++/ggml f32   (deep blue)
    "q8_0": "#2E75B6",       # C++/ggml q8_0  (mid blue)
    "q4_k": "#7FB3D5",       # C++/ggml q4_k  (light blue)
}
GRID = "#d9d9d9"


def load():
    with open(os.path.join(HERE, "results.json")) as fh:
        return json.load(fh)


def label(c):
    return f"{c['engine']}\n{c['quant']}"


def color_for(c):
    return C_TORCH if c["engine"] == "PyTorch" else C_CPP[c["quant"]]


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bars_value_labels(ax, rects, fmt="{:.0f}", dy=3):
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")


def savefig(fig, name):
    p = os.path.join(MEDIA, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def plot_infer_speed(cfgs):
    import numpy as np
    labels = [label(c) for c in cfgs]
    x = np.arange(len(cfgs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    r224 = ax.bar(x - w / 2, [c["infer_ms_224"] for c in cfgs], w,
                  label="@224", color=[color_for(c) for c in cfgs],
                  alpha=0.55, zorder=3, edgecolor="white")
    r504 = ax.bar(x + w / 2, [c["infer_ms_504"] for c in cfgs], w,
                  label="@504 (production)", color=[color_for(c) for c in cfgs],
                  zorder=3, edgecolor="white")
    bars_value_labels(ax, r224)
    bars_value_labels(ax, r504)
    torch504 = cfgs[0]["infer_ms_504"]
    ax.axhline(torch504, ls="--", lw=1.2, color=C_TORCH, zorder=2)
    ax.axhspan(0, torch504, color="#2E75B6", alpha=0.05, zorder=0)
    ax.annotate("PyTorch f32 @504 baseline", (len(cfgs) - 1, torch504),
                xytext=(0, 6), textcoords="offset points", ha="right",
                color=C_TORCH, fontsize=9, fontweight="bold")
    ax.annotate("everything below = faster than PyTorch",
                (0.02, torch504), xytext=(0, -16), textcoords="offset points",
                ha="left", color="#2E75B6", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    style(ax, "Inference latency (DA3-BASE, lower is better)",
          "median ms / image")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_ylim(0, max(c["infer_ms_504"] for c in cfgs) * 1.18)
    fig.text(0.5, -0.02,
             "C++/ggml is FASTER than PyTorch on CPU: 1.20x at f32, 1.31x at q8_0 "
             "(@504); no Python or CUDA at inference.",
             ha="center", fontsize=9, color="#555")
    savefig(fig, "infer_speed.png")


def simple_bar(cfgs, key, title, ylabel, name, fmt="{:.0f}", note=None,
               ann=None):
    import numpy as np
    labels = [label(c) for c in cfgs]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    rects = ax.bar(x, [c[key] for c in cfgs], 0.62,
                   color=[color_for(c) for c in cfgs], zorder=3,
                   edgecolor="white")
    bars_value_labels(ax, rects, fmt=fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    style(ax, title, ylabel)
    ax.set_ylim(0, max(c[key] for c in cfgs) * 1.18)
    if ann:
        ann(ax, rects, cfgs)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#555")
    savefig(fig, name)


def plot_quant_tradeoff(cfgs):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for c in cfgs:
        ax.scatter(c["model_size_mb"], c["infer_ms_504"], s=180,
                   color=color_for(c), zorder=3, edgecolor="white",
                   linewidth=1.5)
        tag = f"{c['engine']} {c['quant']}"
        ax.annotate(tag, (c["model_size_mb"], c["infer_ms_504"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9.5,
                    fontweight="bold")
    style(ax, "Quantization trade-off: model size vs inference latency",
          "infer @504 (median ms)")
    ax.set_xlabel("model size on disk (MB)", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    fig.text(0.5, -0.01,
             "All C++/ggml variants are bit-exact vs the reference (corr=1.0); "
             "q8_0 is fastest (1.31x PyTorch) at 142 MB, q4_k is smallest at 99 MB.",
             ha="center", fontsize=9, color="#555")
    savefig(fig, "quant_tradeoff.png")


def gpu_label(c):
    return f"{c['engine']}\n{c['quant']}"


def plot_gpu_speed(cfgs):
    import numpy as np
    labels = [gpu_label(c) for c in cfgs]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    rects = ax.bar(x, [c["infer_ms_504"] for c in cfgs], 0.62,
                   color=[color_for(c) for c in cfgs], zorder=3,
                   edgecolor="white")
    bars_value_labels(ax, rects, fmt="{:.1f}")
    torch504 = cfgs[0]["infer_ms_504"]
    ax.axhline(torch504, ls="--", lw=1.2, color=C_TORCH, zorder=2)
    ax.annotate("tied with cuDNN", (len(cfgs) - 1, torch504),
                xytext=(0, 18), textcoords="offset points", ha="right",
                color=C_TORCH, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    style(ax, "GPU inference latency (DA3-BASE @504, NVIDIA GB10)",
          "median ms / image (warm)")
    # zoom the y-axis so the ~equal bars read as a clean tie, not a cliff
    ax.set_ylim(0, max(c["infer_ms_504"] for c in cfgs) * 1.30)
    fig.text(0.5, -0.02,
             "On GPU the ggml CUDA path (flash attention, f32) ties PyTorch's tuned "
             "cuDNN at ~47 ms across every quant.",
             ha="center", fontsize=9, color="#555")
    savefig(fig, "gpu_speed.png")


def plot_gpu_load(cfgs):
    import numpy as np
    labels = [gpu_label(c) for c in cfgs]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    rects = ax.bar(x, [c["load_ms"] for c in cfgs], 0.62,
                   color=[color_for(c) for c in cfgs], zorder=3,
                   edgecolor="white")
    bars_value_labels(ax, rects, fmt="{:.0f}")
    torch_load = cfgs[0]["load_ms"]
    ours_f32 = cfgs[1]["load_ms"]
    ax.annotate(f"{torch_load / ours_f32:.1f}x faster",
                (1, ours_f32), xytext=(0, 22), textcoords="offset points",
                ha="center", fontsize=9, color=C_CPP["f32"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    style(ax, "GPU model load time (DA3-BASE, NVIDIA GB10, lower is better)",
          "load (ms)")
    ax.set_ylim(0, max(c["load_ms"] for c in cfgs) * 1.18)
    fig.text(0.5, -0.02,
             "We load 1.75-2.9x faster than PyTorch (mmap'd GGUF, no torch import): "
             "501 ms at f32 down to 306 ms at q4_k vs 879 ms.",
             ha="center", fontsize=9, color="#555")
    savefig(fig, "gpu_load.png")


def plot_gpu_coldstart(cfgs):
    import numpy as np
    labels = [gpu_label(c) for c in cfgs]
    x = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    load = [c["load_ms"] for c in cfgs]
    infer = [c["infer_ms_504"] for c in cfgs]
    r_load = ax.bar(x, load, 0.62, color=[color_for(c) for c in cfgs],
                    zorder=3, edgecolor="white", label="load")
    r_first = ax.bar(x, infer, 0.62, bottom=load, color="#cccccc",
                     zorder=3, edgecolor="white", label="first depth")
    for r, lo, hi in zip(r_load, load, infer):
        total = lo + hi
        ax.annotate(f"{total:.0f}", (r.get_x() + r.get_width() / 2, total),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    style(ax, "GPU cold start: load + first depth (DA3-BASE @504, NVIDIA GB10)",
          "ms (lower is better)")
    ax.set_ylim(0, max(lo + hi for lo, hi in zip(load, infer)) * 1.18)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.text(0.5, -0.02,
             "Cold start (load + first depth) is ~548 ms for ours vs ~926 ms for "
             "PyTorch: the inference is a tie, so the faster load wins the first frame.",
             ha="center", fontsize=9, color="#555")
    savefig(fig, "gpu_coldstart.png")


def main():
    d = load()
    cfgs = d["configs"]

    plot_infer_speed(cfgs)

    def mem_ann(ax, rects, cfgs):
        ax.annotate("~half of PyTorch",
                    (1, cfgs[1]["rss_mb"]), xytext=(0, 22),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=C_CPP["f32"], fontweight="bold")
    simple_bar(cfgs, "rss_mb",
               "Peak memory (RSS, lower is better)", "peak RSS (MB)",
               "memory.png",
               note="C++/ggml uses ~half the RAM of PyTorch "
                    "(614 vs 1328 MB at f32; 320 MB at q4_k).",
               ann=mem_ann)

    def size_ann(ax, rects, cfgs):
        ax.annotate("0.19x size",
                    (3, cfgs[3]["model_size_mb"]), xytext=(0, 22),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=C_CPP["q4_k"], fontweight="bold")
    simple_bar(cfgs, "model_size_mb",
               "Model size on disk (smaller is better)", "size (MB)",
               "model_size.png",
               note="q4_k GGUF is 99 MB: 0.19x of the 516 MB PyTorch checkpoint "
                    "(0.25x of the 393 MB f32 GGUF).",
               ann=size_ann)

    def load_ann(ax, rects, cfgs):
        ax.annotate("~6.7x faster load",
                    (1, cfgs[1]["load_ms"]), xytext=(0, 26),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=C_CPP["f32"], fontweight="bold")
    simple_bar(cfgs, "load_ms",
               "Model load time (lower is better)", "load (ms)",
               "load_time.png",
               note="mmap'd GGUF + no Python/torch import: 25-112 ms vs 749 ms.",
               ann=load_ann)

    plot_quant_tradeoff(cfgs)

    gpu = d.get("gpu")
    if gpu:
        gpu_cfgs = gpu["configs"]
        plot_gpu_speed(gpu_cfgs)
        plot_gpu_load(gpu_cfgs)
        plot_gpu_coldstart(gpu_cfgs)


if __name__ == "__main__":
    main()
