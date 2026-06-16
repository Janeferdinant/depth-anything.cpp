#!/usr/bin/env python3
r"""depth_race — a two-pane "depth race" for depth-anything.cpp, composed frame by
frame with Pillow and encoded with ffmpeg. Modeled on locate-anything.cpp's
image_race.py (panes / fit / progress-bar / encode / gif), but the on-screen
content is a monocular DEPTH map wiping in (turbo colormap), not detection boxes.

The same real photo sits in both panes. Each pane progressively REVEALS its
colorized depth map as a top-to-bottom curtain — a visual stand-in for "computing
depth" — while a thin progress bar under it fills at the REAL measured rate. The
C++ engine is faster, so its curtain/bar finishes first and earns a "FASTER"
badge. The clip holds the finished frame, then fades to a LocalAI end card.

The honest-timing rule (same as image_race): the milliseconds drawn on screen are
the real measured numbers; --dilate only sets PLAYBACK speed, so a sub-second race
is watchable in ~7 s without faking anything. The C++ bar genuinely fills 1.20x
faster than the PyTorch bar.

  .venv/bin/python benchmarks/demo/depth_race.py --gif            # 16:9 hero
  .venv/bin/python benchmarks/demo/depth_race.py --layout square  # 1:1 for social
  .venv/bin/python benchmarks/demo/depth_race.py --photo assets/samples/canyon.jpg

The colorized depth for each photo is precomputed once via the CLI
(da3-cli depth --pfm) and cached under benchmarks/demo/out/.
"""
import argparse, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BG, INK, DIM, GREEN, GOLD = (13, 17, 23), (215, 221, 229), (110, 118, 129), (70, 194, 102), (240, 200, 90)
ACCENTS = {"teal": (62, 200, 224), "slate": (148, 163, 178), "amber": (255, 207, 86),
           "green": (126, 231, 135), "violet": (175, 145, 245), "rose": (244, 130, 150)}
LAYOUTS = {"cols": (1280, 720, "h"), "square": (1080, 1080, "v")}
FPS, RACE_TARGET = 25, 7.0     # frames/s, and the wall-clock length of the race

# Real measured CPU numbers (depth-anything-base, f32, @504 input). The seconds on
# screen are these; --dilate only changes playback speed.
ENGINES = [
    {"label": "PyTorch", "device": "oneDNN CPU", "proc_s": 0.4169, "accent": "slate"},
    {"label": "depth-anything.cpp", "device": "ggml CPU", "proc_s": 0.3464, "accent": "teal"},
]
DEFAULT_PHOTO = "assets/samples/mountains.jpg"
MODEL = "models/depth-anything-base-f32.gguf"


def fontp(bold):
    return f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"
def font(sz, bold=True):
    try: return ImageFont.truetype(fontp(bold), sz)
    except Exception: return ImageFont.load_default()


def read_pfm(path):
    """Minimal PFM reader -> float32 HxW array (top-to-bottom)."""
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        color = header == b"PF"
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
    data = data.reshape((h, w, 3) if color else (h, w))
    if color:
        data = data[..., 0]
    return np.flipud(data)                       # PFM is stored bottom-to-top


def colorize_depth(photo_path, model=MODEL, force=False):
    """Run the CLI to get a depth PFM, turbo-colorize it, cache + return a PIL RGB
    image sized to the source photo."""
    photo = Path(photo_path)
    out_png = HERE / "out" / f"{photo.stem}_depth.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.exists() and not force:
        return Image.open(out_png).convert("RGB")
    with tempfile.TemporaryDirectory() as tmp:
        pfm = Path(tmp) / "d.pfm"
        subprocess.run([str(ROOT / "build/examples/cli/da3-cli"), "depth",
                        "--model", str(ROOT / model), "--input", str(photo),
                        "--pfm", str(pfm)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = read_pfm(pfm)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    n = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
    rgb = (cm.turbo(n)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    src = Image.open(photo).convert("RGB")
    img = img.resize(src.size, Image.LANCZOS)
    img.save(out_png)
    return img


def fit(img, box_w, box_h):
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def panes(W, H, orient, top):
    if orient == "h":
        pw = (W - 80) // 2
        return [(40, top, pw, H - top - 54), (40 + pw, top, pw, H - top - 54)]
    ph = (H - top - 54) // 2
    return [(60, top, W - 120, ph), (60, top + ph + 20, W - 120, ph)]


def draw_pane(cv, rect, photo, depth, eng, frac, winner):
    d = ImageDraw.Draw(cv); ox, oy, pw, ph = rect; accent = eng["_c"]
    done = frac >= 1.0
    iw, ih = photo.size
    ix, iy = ox + (pw - iw) // 2, oy + 6
    # reveal: a top-to-bottom curtain of the colorized depth over the photo
    reveal_h = int(round(ih * min(1.0, frac)))
    comp = photo.copy()
    if reveal_h > 0:
        comp.paste(depth.crop((0, 0, iw, reveal_h)), (0, 0))
    cv.paste(comp, (ix, iy))
    if 0 < reveal_h < ih:                         # bright scan edge of the wipe
        ey = iy + reveal_h
        d.line([ix, ey, ix + iw, ey], fill=(245, 248, 252), width=3)
        d.line([ix, ey + 2, ix + iw, ey + 2], fill=accent, width=1)
    d.rectangle([ix - 1, iy - 1, ix + iw, iy + ih], outline=accent, width=2)
    # "FASTER" badge for the winner once finished
    if done and winner:
        bf = font(17); txt = "✓ FASTER"
        tw = d.textlength(txt, font=bf); bx, by = ix + iw - tw - 22, iy + 10
        d.rounded_rectangle([bx, by, bx + tw + 16, by + 28], 8, fill=GREEN)
        d.text((bx + 8, by + 5), txt, fill=BG, font=bf)
    # label + device
    cy = iy + ih + 12; fs = font(20); ft = font(16, False)
    d.text((ix, cy), eng["label"], fill=accent, font=fs)
    d.text((ix + d.textlength(eng["label"], font=fs) + 10, cy + 2), eng["device"], fill=DIM, font=ft)
    # progress bar
    by = cy + 30
    d.rounded_rectangle([ix, by, ix + iw, by + 8], 4, fill=(34, 41, 50))
    d.rounded_rectangle([ix, by, ix + int(iw * min(1.0, frac)), by + 8], 4, fill=accent)
    # status line (sub-second -> milliseconds)
    sy = by + 15; ms = eng["proc_s"] * 1000
    if done:
        s = f"✓ {ms:.1f} ms"; d.text((ix, sy), s, fill=INK, font=fs)
        if winner:
            d.text((ix + d.textlength(s, font=fs) + 16, sy), "★ fastest", fill=GOLD, font=fs)
    else:
        d.text((ix, sy), f"▸ {min(frac, 1.0) * ms:.0f} ms", fill=accent, font=fs)


def frame(W, H, orient, photo_panes, depth_panes, t_real, note):
    cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
    fh = font(max(22, W // 44)); ft = font(16, False)
    a, b = ENGINES
    d.text((40, 26), a["label"], fill=a["_c"], font=fh)
    x = 40 + d.textlength(a["label"], font=fh)
    d.text((x + 12, 32), "vs", fill=DIM, font=ft)
    d.text((x + 44, 26), b["label"], fill=b["_c"], font=fh)
    if note:
        d.text((W - 40 - d.textlength(note, font=ft), 32), note, fill=DIM, font=ft)
    d.line([40, 74, W - 40, 74], fill=(34, 43, 52), width=1)
    top = 96
    rects = panes(W, H, orient, top)
    fastest = min(ENGINES, key=lambda e: e["proc_s"])
    for r, e, ph, dp in zip(rects, ENGINES, photo_panes, depth_panes):
        draw_pane(cv, r, ph, dp, e, min(1.0, t_real / e["proc_s"]), e is fastest)
    return cv


def _ctext(d, W, y, text, fill, fnt):
    d.text(((W - d.textlength(text, font=fnt)) / 2, y), text, fill=fill, font=fnt)


def end_card(W, H):
    """LocalAI finish CTA: logo + 'from the LocalAI team' + the headline result."""
    cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
    a, b = sorted(ENGINES, key=lambda e: e["proc_s"])
    ratio = b["proc_s"] / a["proc_s"]
    # logo, centered near the top
    logo = Image.open(ROOT / "assets/localai_logo.png").convert("RGBA")
    lh = int(H * 0.30); lw = int(logo.width * lh / logo.height)
    logo = logo.resize((lw, lh), Image.LANCZOS)
    ly = int(H * 0.085)
    cv.paste(logo, ((W - lw) // 2, ly), logo)
    # accent divider
    teal = ACCENTS["teal"]
    dy = ly + lh + int(H * 0.045)
    d.rectangle([(W - 120) // 2, dy, (W + 120) // 2, dy + 3], fill=teal)
    # title + headline + subline
    _ctext(d, W, dy + int(H * 0.025), "from the LocalAI team", INK, font(max(22, W // 50)))
    big = font(max(34, W // 28))
    _ctext(d, W, dy + int(H * 0.075), f"{ratio:.2f}x faster than PyTorch on CPU", teal, big)
    _ctext(d, W, dy + int(H * 0.165), "bit-exact, half the RAM, no Python", DIM, font(max(18, W // 62)))
    # footer links
    fl = font(max(16, W // 66), False)
    _ctext(d, W, H - int(H * 0.115), "github.com/mudler/depth-anything.cpp", teal, fl)
    _ctext(d, W, H - int(H * 0.065), "huggingface.co/mudler/depth-anything.cpp-gguf", DIM, fl)
    return cv


def encode(frames, out, orient, gif):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, fr in enumerate(frames):
            fr.save(tmp / f"f{i:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(tmp / "f%05d.png"), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(out)], check=True)
        if gif:
            pal = tmp / "pal.png"; gw = 760 if orient == "h" else 600
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                            "-vf", f"fps=14,scale={gw}:-1:flags=lanczos,palettegen=stats_mode=diff", str(pal)], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(pal),
                            "-lavfi", f"fps=14,scale={gw}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                            str(out.with_suffix(".gif"))], check=True)


def fade(a, b, t):
    return Image.blend(a, b, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", default=DEFAULT_PHOTO)
    ap.add_argument("--out", default=str(ROOT / "benchmarks/media/depth_race.mp4"))
    ap.add_argument("--layout", choices=list(LAYOUTS), default="cols")
    ap.add_argument("--dilate", type=float, default=0.0, help="playback factor; 0 = auto-fit race to ~7 s")
    ap.add_argument("--note", default="same photo · same CPU · turbo depth")
    ap.add_argument("--gif", action="store_true", help="also write a .gif next to the mp4")
    ap.add_argument("--no-card", action="store_true", help="omit the LocalAI end card")
    ap.add_argument("--force", action="store_true", help="recompute the cached depth map")
    a = ap.parse_args()
    W, H, orient = LAYOUTS[a.layout]
    for e in ENGINES:
        e["_c"] = ACCENTS[e["accent"]]

    photo = Image.open(ROOT / a.photo if not Path(a.photo).is_absolute() else a.photo).convert("RGB")
    depth = colorize_depth(ROOT / a.photo if not Path(a.photo).is_absolute() else a.photo, force=a.force)

    rects = panes(W, H, orient, 96)
    photo_panes, depth_panes = [], []
    for r in rects:
        ph = fit(photo, r[2] - 20, r[3] - 90)
        photo_panes.append(ph)
        depth_panes.append(depth.resize(ph.size, Image.LANCZOS))

    proc_max = max(e["proc_s"] for e in ENGINES)
    dilate = a.dilate if a.dilate > 0 else RACE_TARGET / proc_max
    wall = proc_max * dilate

    frames = []
    for i in range(int(wall * FPS) + 1):
        frames.append(frame(W, H, orient, photo_panes, depth_panes, (i / FPS) / dilate, a.note))
    last = frame(W, H, orient, photo_panes, depth_panes, wall / dilate, a.note)
    frames += [last] * int(1.0 * FPS)             # hold the finished race
    if not a.no_card:
        card = end_card(W, H)
        for i in range(int(0.4 * FPS)):           # crossfade into the card
            frames.append(fade(last, card, (i + 1) / int(0.4 * FPS)))
        frames += [card] * int(3.5 * FPS)         # hold the CTA
    encode(frames, a.out, orient, a.gif)
    print("wrote", a.out, ("+ gif" if a.gif else ""))


if __name__ == "__main__":
    main()
