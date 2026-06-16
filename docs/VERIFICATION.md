# Verification and parity

depth-anything.cpp is verified **numerically equal to the original Depth Anything 3
PyTorch model**, component by component, not just on the final output. Every stage is
gated against tensors dumped from the reference model, and the end-to-end output is
checked against the genuine `net()` forward on real (non-fixture) images.

The gates live in the ctest suite (`-DDA_BUILD_TESTS=ON`, run `ctest --test-dir build`)
and the `scripts/e2e_*.py` end-to-end checks (which run the real PyTorch model and
compare). `corr` is the Pearson correlation of the flattened output vs the f32
reference; `max|d|` is the max absolute difference.

## End-to-end parity (C++/ggml vs original PyTorch)

| path | result | notes |
|------|--------|-------|
| **Depth** | max\|d\|=9.5e-7, **corr=1.000000** | f32 rounding noise; gate corr>0.999, max\|d\|<5e-3 |
| **Camera pose** | extrinsics max\|d\|=4.5e-8, intrinsics max\|d\|=2.6e-4 | focal magnitudes ~244-347 px |
| **Multi-view depth+pose** | corr=1.000000 | two structured views, cross-view attention |
| **3D Gaussian reconstruction** (GIANT) | gated vs reference | GaussianAdapter -> world-space Gaussians + .ply |
| **Nested metric depth** | gated vs reference | two-branch alignment (anyview + metric) |
| **Ray-based pose** (`use_ray_pose`) | rotation max\|d\|=2.4e-7, intrinsics rel 2.3e-6 | aux ray head bit-exact; solver fed identical RANSAC indices |
| **Native-resolution e2e** | max\|d\|=1.37e-6, **corr=1.000000** | raw arbitrary-resolution photo, bit-exact cv2 resize + the real `net.head` |

The resize is bit-exact vs OpenCV (`computeResizeAreaTab` / INTER_AREA + INTER_CUBIC
ported exactly), and the f32 forward is parity-preserving, so end-to-end depth lands at
f32-noise level (corr=1.0).

## Quantization accuracy

Quantized GGUFs preserve depth and pose relative to the f32 reference
(`tests/test_quantize_accuracy.cpp`):

| model | size | depth max\|d\| vs f32 | depth corr | ext max\|d\| |
|-------|------|----------------------:|-----------:|-------------:|
| f32   | 393 MB | 0 (exact)   | 1.000000 | 0 (exact) |
| q8_0  | 142 MB | 1.9e-3      | 0.999979 | 2.0e-3 |
| q4_k  |  99 MB | 1.9e-2      | 0.998579 | 1.8e-2 |

q8_0 is near-lossless; q4_k stays above 0.998 correlation (well above the 0.99 floor)
at the smallest size.

## Reproducing

```sh
cmake -B build -DDA_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build            # 37 component + e2e gates

# end-to-end vs the genuine PyTorch model (needs the conversion venv)
python scripts/e2e_verify_native.py --model-dir models/DA3-BASE --gguf models/depth-anything-base-f32.gguf
```
