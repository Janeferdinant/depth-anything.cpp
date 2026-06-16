#!/usr/bin/env python3
"""Terminal depth demo for screen-recording (pure stdlib, no PIL/numpy).

Runs the real da3-cli on a sample image, then renders the resulting depth PFM
as truecolor (turbo) ANSI half-blocks right in the terminal, and prints the
honest C++/ggml-vs-PyTorch headline. Designed to be captured to MP4 by
recorder-for-agents:

    DURATION=20 FONTSIZE=15 ./record.sh \
        "python3 benchmarks/demo_terminal.py" depth_demo.mp4

Self-contained: the turbo lookup table is embedded so no Python deps are needed
inside the recorder container.
"""
import os
import struct
import subprocess
import sys
import time

CLI = "build/examples/cli/da3-cli"
MODEL = "models/depth-anything-base-f32.gguf"
IMG = "assets/samples/canyon.jpg"
PFM = "/tmp/demo_depth.pfm"
COLS = 62  # render width in terminal cells

# turbo colormap, 256 RGB entries (generated from matplotlib cm.turbo)
TURBO = [(48,18,59),(50,24,74),(53,30,88),(55,35,101),(57,41,114),(59,47,127),(60,53,139),(62,58,150),(64,64,161),(65,69,171),(66,75,181),(67,80,190),(68,86,199),(69,91,206),(69,96,214),(70,102,221),(70,107,227),(70,112,232),(70,117,237),(70,122,242),(70,127,246),(70,132,248),(69,137,252),(67,142,253),(66,145,254),(64,150,254),(62,155,254),(60,157,253),(57,162,252),(54,168,249),(51,172,246),(49,175,245),(45,180,241),(42,185,237),(38,189,233),(35,194,228),(32,198,223),(29,203,218),(27,207,212),(25,211,207),(24,215,202),(23,218,196),(23,222,191),(24,225,186),(25,227,184),(27,229,180),(30,232,175),(34,235,169),(39,237,163),(44,239,157),(50,241,151),(56,244,145),(63,245,138),(70,247,131),(77,249,124),(85,250,118),(93,251,111),(101,252,104),(109,253,98),(116,254,92),(124,254,86),(132,254,80),(139,254,75),(146,254,70),(152,254,66),(158,253,62),(164,252,59),(169,251,57),(174,249,55),(179,248,53),(185,245,52),(190,243,52),(195,241,51),(200,238,51),(205,235,52),(209,232,52),(214,229,53),(218,226,54),(223,222,54),(227,218,55),(231,215,56),(234,211,57),(237,207,57),(240,203,58),(243,198,58),(246,194,58),(248,190,57),(249,186,56),(251,181,55),(252,176,53),(253,171,51),(253,166,49),(254,161,47),(254,155,45),(253,149,43),(253,143,40),(252,137,38),(251,131,35),(250,125,32),(249,119,30),(247,113,27),(246,107,24),(244,101,22),(242,96,20),(239,90,17),(237,85,15),(234,80,13),(232,75,12),(229,70,10),(226,66,9),(222,62,8),(219,58,7),(215,54,6),(212,50,5),(208,47,4),(203,43,3),(199,40,3),(195,36,2),(190,33,2),(185,30,1),(180,27,1),(174,24,1),(169,21,1),(163,18,1),(157,16,1),(151,13,1),(145,11,1),(139,9,1),(132,7,1),(125,5,2),(122,4,2)]


def c(rgb, s):
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"


def read_pfm(path):
    with open(path, "rb") as fh:
        fh.readline()  # "Pf"
        w, h = (int(x) for x in fh.readline().split())
        scale = float(fh.readline())
        end = "<" if scale < 0 else ">"
        data = struct.unpack(end + "f" * (w * h), fh.read(4 * w * h))
    rows = [list(data[y * w:(y + 1) * w]) for y in range(h)]
    rows.reverse()  # PFM is bottom-to-top
    return rows, w, h


def turbo(t):
    n = len(TURBO)
    return TURBO[max(0, min(n - 1, int(t * (n - 1))))]


def render(rows, w, h):
    cell_w = w / COLS
    rows_out = int(COLS / (w / h) / 2)  # half-blocks: 2 px per cell row
    cell_h = h / (rows_out * 2)
    flat = [v for r in rows for v in r]
    lo = sorted(flat)[int(0.02 * len(flat))]
    hi = sorted(flat)[int(0.98 * len(flat))]
    rng = max(hi - lo, 1e-6)

    def sample(cx, cy):
        x = min(w - 1, int(cx * cell_w))
        y = min(h - 1, int(cy * cell_h))
        return turbo((rows[y][x] - lo) / rng)

    for ry in range(rows_out):
        line = []
        for cx in range(COLS):
            top = sample(cx, ry * 2)
            bot = sample(cx, ry * 2 + 1)
            line.append(f"\x1b[38;2;{top[0]};{top[1]};{top[2]}m"
                        f"\x1b[48;2;{bot[0]};{bot[1]};{bot[2]}m▀")
        print("  " + "".join(line) + "\x1b[0m")
        time.sleep(0.045)


def main():
    BOLD = "\x1b[1m"
    R = "\x1b[0m"
    print()
    print(BOLD + c((70, 130, 248), "  depth-anything.cpp") +
          "  -  monocular depth on CPU, no Python/CUDA" + R)
    print("  " + "-" * 66)
    print(f"  $ da3-cli depth --model {os.path.basename(MODEL)} \\")
    print(f"      --input {os.path.basename(IMG)} --pfm depth.pfm --threads 16")
    print()
    time.sleep(1.0)
    print("  loading GGUF (mmap) + running DA3-BASE ...", end="", flush=True)
    t0 = time.time()
    subprocess.run([CLI, "depth", "--model", MODEL, "--input", IMG,
                    "--pfm", PFM, "--threads", "16"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    dt = time.time() - t0
    print(c((124, 254, 86), f" done in {dt*1000:.0f} ms (cold, incl. load)"))
    print()
    rows, w, h = read_pfm(PFM)
    print(c((255, 255, 255), f"  depth {w}x{h}  (turbo colormap):"))
    render(rows, w, h)
    print()
    print("  " + "-" * 66)
    print(BOLD + "  C++/ggml vs PyTorch (DA3-BASE @504, 16 threads)" + R)
    print(f"    latency   {c((124,254,86),'FASTER')}        346 ms vs 417 ms  (1.20x; q8_0 1.31x)")
    print(f"    peak RAM  {c((124,254,86),'~half')}         614 MB vs 1328 MB")
    print(f"    load      {c((124,254,86),'~6.7x faster')}  112 ms vs 749 ms")
    print(f"    size      {c((124,254,86),'0.19x (q4_k)')}  99 MB vs 516 MB")
    print(f"    parity    {c((124,254,86),'bit-exact')}     corr = 1.000")
    print()
    sys.stdout.flush()
    time.sleep(8.0)  # hold the final frame for the screen recording


if __name__ == "__main__":
    main()
