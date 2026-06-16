#!/usr/bin/env python3
"""Render the release demo visuals for depth-anything.cpp.

- depth_demo.png : input photo | colorized depth (turbo) side-by-side panels.
- sky_demo.png   : (optional) mono model sky mask, if the PFMs are present.
- timing_demo.png/.gif : honest C++ vs PyTorch @504 timing comparison.

Pure-offline: matplotlib + numpy + PIL only, no network. Depth PFMs are
produced by the CLI (see benchmarks/BENCHMARK.md / the coordinator's recipe);
this script only reads .pfm + .png inputs and renders the figures.

Run:  .venv/bin/python benchmarks/make_demo.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MEDIA = os.path.join(HERE, "media")
os.makedirs(MEDIA, exist_ok=True)

C_TORCH = "#E07B39"
C_CPP = "#1F4E79"


def read_pfm(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode("ascii").strip()
        color = header == "PF"
        w, h = (int(x) for x in fh.readline().split())
        scale = float(fh.readline().strip())
        endian = "<" if scale < 0 else ">"
        ch = 3 if color else 1
        data = np.fromfile(fh, endian + "f", w * h * ch)
        data = data.reshape((h, w, ch) if color else (h, w))
        return np.flipud(data)  # PFM is bottom-to-top


def colorize(depth, cmap="turbo", mask=None):
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    lo, hi = np.percentile(d[finite], 2), np.percentile(d[finite], 98)
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[mask] = (20, 24, 30)
    return Image.fromarray(rgb)


def panel(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def depth_demo():
    # Real photos -> input | colorized-depth, rendered from the CLI's PFM output.
    # PFMs are produced by:
    #   for f in mountains canyon street desk; do
    #     build/examples/cli/da3-cli depth --model models/depth-anything-base-f32.gguf \
    #       --input assets/samples/$f.jpg --pfm /tmp/dademo/$f.pfm --threads 16
    #   done
    samples = [
        ("assets/samples/mountains.jpg", "/tmp/dademo/mountains.pfm", "mountains"),
        ("assets/samples/canyon.jpg",    "/tmp/dademo/canyon.pfm",    "canyon"),
        ("assets/samples/street.jpg",    "/tmp/dademo/street.pfm",    "street"),
        ("assets/samples/desk.jpg",      "/tmp/dademo/desk.pfm",      "desk"),
    ]
    rows = [(i, p, n) for i, p, n in samples
            if os.path.exists(os.path.join(ROOT, i)) and os.path.exists(p)]
    n = len(rows)
    if n == 0:
        print("skip depth_demo (no sample PFMs; run the CLI recipe above)")
        return
    fig, axes = plt.subplots(n, 2, figsize=(9.4, 3.05 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (img_rel, pfm, name) in enumerate(rows):
        img = Image.open(os.path.join(ROOT, img_rel)).convert("RGB")
        depth = read_pfm(pfm)
        dcol = colorize(depth).resize(img.size, Image.BILINEAR)
        panel(axes[r, 0], img, "input")
        panel(axes[r, 1], dcol, "depth (DA3-BASE, C++/ggml, turbo)")
    fig.suptitle("depth-anything.cpp - monocular metric depth on real photos (CPU)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = os.path.join(MEDIA, "depth_demo.png")
    fig.savefig(p, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def sky_demo():
    inp = os.path.join(ROOT, "dumps/mono_input.png")
    dpfm = os.path.join(ROOT, "dumps/mono_cpp.pfm")
    spfm = os.path.join(ROOT, "dumps/mono_sky_cpp.pfm")
    if not all(os.path.exists(p) for p in (inp, dpfm, spfm)):
        print("skip sky_demo (mono PFMs not present)")
        return
    img = Image.open(inp).convert("RGB")
    depth = read_pfm(dpfm)
    sky = read_pfm(spfm)
    sky_mask = sky > 0.5  # mono sky logits/prob > 0.5 = sky
    dcol = colorize(depth).resize(img.size, Image.BILINEAR)
    sky_rs = np.array(Image.fromarray((sky_mask * 255).astype(np.uint8))
                      .resize(img.size, Image.NEAREST)) > 127
    overlay = np.array(img).copy()
    overlay[sky_rs] = (0.4 * overlay[sky_rs] +
                       0.6 * np.array([90, 170, 255])).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.2))
    panel(axes[0], img, "input")
    panel(axes[1], dcol, "depth (mono-large, turbo)")
    panel(axes[2], Image.fromarray(overlay), "sky mask (blue overlay)")
    fig.suptitle("depth-anything.cpp - mono-large: depth + sky mask",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(MEDIA, "sky_demo.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# --- timing race: C++/ggml finishes the forward FIRST (faster on CPU) --------
# Both bars advance at the same wall-clock rate (px per ms); the shorter one
# (C++) reaches its finish line first. Honest: the real measured @504 ms.
TORCH_MS, CPP_MS = 416.9, 346.4      # infer @504 (f32), sustained
TORCH_RSS, CPP_RSS = 1328, 614       # peak RSS MB
SPEEDUP = TORCH_MS / CPP_MS          # 1.20x


def _timing_axes(ax, t_ms):
    """Race state at elapsed wall-clock t_ms: each bar fills to min(t, its ms)."""
    ax.clear()
    span = TORCH_MS * 1.18
    rows = [("PyTorch f32", TORCH_MS, C_TORCH),
            ("C++/ggml f32", CPP_MS, C_CPP)]
    y = [1, 0]
    for (name, ms, color), yi in zip(rows, y):
        ax.barh(yi, span, height=0.5, color="#eee", zorder=1)
        cur = min(t_ms, ms)
        ax.barh(yi, cur, height=0.5, color=color, zorder=2)
        # finish line + DONE flag once this engine has crossed it
        ax.axvline(ms, ymin=0, ymax=1, color=color, lw=1.0, ls=":",
                   alpha=0.6, zorder=1)
        if t_ms >= ms - 1e-6:
            tag = "WINNER" if ms == CPP_MS else "done"
            ax.text(ms + span * 0.012, yi, f"{ms:.0f} ms  {tag}",
                    va="center", ha="left", fontsize=12, fontweight="bold",
                    color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(["PyTorch f32\n1328 MB RAM", "C++/ggml f32\n614 MB RAM"],
                       fontsize=11)
    ax.set_xlim(0, span)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("inference latency @504 (ms) - lower is better, first to finish wins",
                  fontsize=10.5)
    ax.set_title("C++/ggml beats PyTorch on CPU (DA3-BASE @504)",
                 fontsize=14, fontweight="bold", pad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e3e3e3", zorder=0)
    ax.set_axisbelow(True)


def timing_png():
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    _timing_axes(ax, TORCH_MS)
    fig.text(0.5, -0.04,
             f"C++/ggml f32 finishes the @504 forward first: {CPP_MS:.0f} vs "
             f"{TORCH_MS:.0f} ms ({SPEEDUP:.2f}x faster), using ~half the RAM "
             f"(614 vs 1328 MB),\nloading ~6.7x faster, no Python/CUDA. q8_0 is "
             f"1.31x at 363 MB RSS. Bit-exact (corr=1.0).",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    p = os.path.join(MEDIA, "timing_demo.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def timing_gif():
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print("skip gif:", e)
        return
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.84, bottom=0.20)
    run_frames = 40        # frames spent racing to PyTorch's finish
    hold = 22              # hold the final frame (C++ already won)

    def update(i):
        if i < run_frames:
            t_ms = TORCH_MS * (i / (run_frames - 1))
        else:
            t_ms = TORCH_MS
        _timing_axes(ax, t_ms)

    anim = FuncAnimation(fig, update, frames=run_frames + hold, interval=40)
    p = os.path.join(MEDIA, "timing_demo.gif")
    anim.save(p, writer=PillowWriter(fps=24))
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    depth_demo()
    # sky_demo() is only meaningful on real outdoor photos (the bundled sample
    # is a synthetic test image with no real sky), so it is not rendered by
    # default. Call it manually with mono PFMs from an outdoor photo.
    timing_png()
    timing_gif()
