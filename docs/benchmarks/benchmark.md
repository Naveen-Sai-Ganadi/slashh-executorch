# StressNet benchmarks — NPU evidence + accuracy

## 1. On-device NPU profile (the 40% evidence) ⭐

Qualcomm AI Hub Workbench, **Samsung Galaxy S25 Ultra** (Snapdragon 8 Elite for
Galaxy, `sm8750-ac`, Android 15) — the canonical device from plan §8.3. Real
trained weights (`assets/stress_model.pt`), INT8, QNN context binary.

| Metric | Value |
|---|---|
| **NPU delegation** | **18 / 18 layers = 100% NPU** (0 CPU, 0 GPU) |
| **Median inference** | **0.060 ms** (60 µs) per [1,1,64,301] window |
| Min inference | 0.052 ms |
| Peak memory | ~105 MB |
| Deployable artifact | `assets/stress_model_qnn.bin` (80 KB) |

Reproduce: `python -m model.aihub_profile --submit --weights assets/stress_model.pt`

AI Hub jobs:
- compile→ONNX: https://workbench.aihub.qualcomm.com/jobs/jp86jk48g/
- INT8 quantize: https://workbench.aihub.qualcomm.com/jobs/j5qz4dmm5/
- compile→QNN: https://workbench.aihub.qualcomm.com/jobs/jglowq1lg/
- **profile (NPU)**: https://workbench.aihub.qualcomm.com/jobs/jgd86mzl5/

## 2. Accuracy (real data)

Trained on **RAVDESS**, plan §13 mapping (angry/fearful → stressed, neutral/calm
→ calm), **speaker-independent** split (val actors 21–24 unseen in training).

| Precision | Val accuracy | Balanced accuracy |
|---|---|---|
| FP32 | 95.5% | **95.8%** |
| w8a8 (PT2E, local) | — | **96.9%** (no drop) |

Reproduce: `python -m model.train --ravdess data --epochs 40`

## 3. Host CPU baseline (XNNPACK) — for the CPU-vs-NPU A/B slide

- host: `macOS-26.5.1-arm64-arm-64bit` · torch `2.11.0` · threads 8

| variant | channels | dtype | eager latency (ms) | pte latency (ms) | speedup | .pt size (KB) | .pte size (KB) | parity max abs err |
|---|---|---|---|---|---|---|---|---|
| small | (8, 16, 32) | fp32 | 0.630 | 0.647 | 0.97× | 30.5 | 29.9 | 5.96e-08 |
| base | (16, 32, 64) | fp32 | 0.777 | 0.799 | 0.97× | 99.2 | 97.9 | 5.96e-08 |
| base-int8 | (16, 32, 64) | int8 | 0.729 | 0.650 | 1.12× | 99.2 | 33.8 | 2.30e-04 |

_Latency is mean wall-clock per 1×[1,1,64,301] window._

**The A/B headline for the slide:** ~0.78 ms on CPU (XNNPACK) vs **0.060 ms on the
Hexagon NPU** — a ~13× per-window speedup, at 100% NPU delegation, on the exact
demo silicon.
