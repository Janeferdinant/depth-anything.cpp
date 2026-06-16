# 3D Export: `.glb` and COLMAP

depth-anything.cpp can turn a single image's predicted depth + camera pose into
two standard 3D artifacts, with **no Python and no third-party libraries**
(trimesh / pycolmap are not used — both formats are serialized by hand in C++):

- **glTF-2.0 binary (`.glb`)** — a point cloud (`POINTS` primitive) plus optional
  camera-frustum wireframes (`LINES` primitive). Opens in Blender, three.js, and
  any glTF viewer.
- **COLMAP sparse model** — `cameras` / `images` / `points3D` in either the
  little-endian `.bin` layout (default, matches `pycolmap`'s
  `reconstruction.write`) or the `.txt` variant.

The geometry mirrors the reference exporters
(`depth_anything_3/utils/export/{glb,colmap}.py`) and is parity-verified
byte-for-byte (see [Parity](#parity)).

## Pipeline

For a single image the export path runs the **native-resolution** depth+pose
pipeline (`Engine::depth_pose_native`):

1. Preprocess the image with the real DA3 resize policy to `(W,H)` (long side
   ≈ `img_resize_target`, both multiples of `patch_size`).
2. One backbone pass → DualDPT depth head (`depth`, `conf`) + camera head
   (`ext` 3×4 row-major world-to-camera, `intr` 3×3 row-major) at processed size.
3. Build the 4×4 extrinsic by appending `[0,0,0,1]` to the 3×4.
4. Back-project each valid pixel into a shared world frame (`reconstruct.cpp`),
   colored by the **processed-resolution RGB uint8** (the resized pixels *before*
   mean/std normalization — captured directly from the preprocess step so the
   colors are guaranteed consistent with the model input).

A pixel is valid when `isfinite(d) && d>0 && conf>=conf_thr`.

### Determinism (downsampling disabled)

The reference `.glb` exporter randomly downsamples points with
`np.random.choice` (nondeterministic). For reproducible, parity-checkable output
we **keep all valid points** (`GlbOptions::num_max_points` defaults to 1,000,000,
which covers a full-resolution single frame). COLMAP export never downsamples.

### Confidence thresholds (faithful to the reference)

- **COLMAP:** `conf_thr = percentile(conf, 40)` over all frames (numpy linear
  interpolation).
- **GLB** (`get_conf_thresh`, params `conf_thresh=1.05`,
  `conf_thresh_percentile=40`, `ensure_thresh_percentile=90`):
  `lower = pct(conf,40)`, `upper = pct(conf,90)`,
  `thr = min(max(1.05, lower), upper)`. (Our depth path has no sky mask, so all
  confidences are used.)

## CLI

```
da3-cli depth --model <gguf> --input <img> [--glb <out.glb>] [--colmap <out_dir>] [--colmap-txt <out_dir>]
```

| Flag | Effect |
|------|--------|
| `--glb <out.glb>` | Write a glTF-2.0 binary point cloud (+ camera frustum). |
| `--colmap <out_dir>` | Write a COLMAP model as `cameras.bin` / `images.bin` / `points3D.bin`. |
| `--colmap-txt <out_dir>` | Same, but the `.txt` variant. |

`--glb`/`--colmap` support a **single** `--input` only. They can be combined with
`--pfm` / `--png` (the depth map is also written). If the model cannot produce a
camera pose the command exits with a clear error.

### Example

```sh
da3-cli depth --model models/depth-anything-base-f32.gguf \
              --input photo.png \
              --glb /tmp/scene.glb \
              --colmap /tmp/colmap_out
```

Produces `/tmp/scene.glb` (point cloud) and
`/tmp/colmap_out/{cameras,images,points3D}.bin`. The COLMAP model round-trips
through `pycolmap` / `read_write_model.py`.

## C API

```c
#include "da_capi.h"

da_ctx* ctx = da_capi_load("model.gguf", /*n_threads*/ 4);

/* glTF-2.0 binary point cloud. Returns 0 ok, -1 error. */
int rc1 = da_capi_export_glb(ctx, "photo.png", "/tmp/scene.glb");

/* COLMAP model. binary != 0 => .bin (default), 0 => .txt. Returns 0 ok, -1 error. */
int rc2 = da_capi_export_colmap(ctx, "photo.png", "/tmp/colmap_out", /*binary*/ 1);

if (rc1 != 0 || rc2 != 0) fprintf(stderr, "%s\n", da_capi_last_error(ctx));
da_capi_free(ctx);
```

`da_capi_abi_version()` returns **2** (the export wrappers were added in this
version).

## File formats

### glTF-2.0 binary (`.glb`)

12-byte header + JSON chunk (padded with `0x20`) + BIN chunk (padded with
`0x00`), every `bufferView` 4-byte aligned. Accessors: `POSITION` as `VEC3`
float, `COLOR_0` as normalized unsigned-byte `VEC4` (alpha 255). The point cloud
is aligned to the first camera in glTF coordinates and centered on the per-axis
median of the points. Optional camera frustums are emitted as a `LINES`
primitive.

### COLMAP (little-endian `.bin`)

- **cameras.bin:** `uint64 num`; per camera `int32 id, int32 model_id(=1 PINHOLE),
  uint64 width, uint64 height`, then `4×float64` params `[fx,fy,cx,cy]`.
  Intrinsics are rescaled to the **original** image size
  (`fx,cx *= orig_w/W`, `fy,cy *= orig_h/H`); width/height are the original size.
- **images.bin:** `uint64 num`; per image `int32 id`, `4×float64 qvec(qw,qx,qy,qz)`,
  `3×float64 tvec`, `int32 camera_id`, NUL-terminated name, `uint64 num_pts2D`,
  then per 2D point `float64 x, float64 y, int64 point3D_id`.
  `qvec = rotmat2qvec(R = ext[:3,:3])`, `tvec = ext[:3,3]`.
- **points3D.bin:** `uint64 num`; per point `uint64 id, 3×float64 xyz, 3×uint8 rgb,
  float64 error(=0)`, `uint64 track_len`, then per track elem `int32 image_id,
  int32 point2D_idx`. Point3D ids are `1..num_points` in back-projection order.

The `.txt` variants follow `read_write_model.py`'s `write_*_text` layout.

## Parity

The exporter geometry + byte encoding are verified against a faithful numpy
re-implementation of the reference math (the reference modules are not imported
at test time):

- `scripts/parity_glb.py` — builds expected aligned points/colors, runs the
  `glb_parity_dump` harness, parses the `.glb` `POSITION`+`COLOR_0` accessors,
  asserts sorted point/color sets match (`max|d| < 1e-4`, colors exact).
- `scripts/parity_colmap.py` — replicates `colmap.py`'s field math (intrinsic
  rescale, `rotmat2qvec`), runs the `colmap_parity_dump` harness, reads the
  `.bin` with `read_write_model.py`, asserts cameras / image qvec+tvec /
  points3D xyz+rgb match. Cross-checks with `pycolmap` if importable.

Both currently report `PASS`. The model itself is independently parity-verified
(engine e2e corr = 1.0); these gates cover only exporter geometry + encoding.
