# INT8 calibration A/B — clean vs noise-aware

The shipped deployable is the INT8 `.pte`; PT2E picks its integer scales from a **calibration** batch. The same trained model is quantized two ways — calibrated on clean audio (today's default) vs on noise-augmented audio from the training SNR pool — then both INT8 runtimes and the eager **fp32** model are swept over *identical* noisy inputs, so the only variable is the calibration distribution.

- fp32 reliable down to: **-5 dB** (accuracy ≥ 0.80)
- **No difference** — clean and noise-aware calibration hold the same reliable floor. At this size the quantizer's scales already cover the noisy activations, so the simpler clean calibration is sufficient; keep it. Recommended calibration: **`clean`** (reliable to -5 dB).

| calibration | reliable floor | mean |Δ| vs fp32 | preserves fp32 floor |
|---|---|---|---|
| `clean` ⭐ | -5 dB | 0.000 | yes |
| `noise-aware` | -5 dB | 0.000 | yes |

### Per-SNR accuracy

| SNR | fp32 | INT8 (clean calib) | INT8 (noise-aware calib) |
|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 |
| 20 dB | 1.000 | 1.000 | 1.000 |
| 10 dB | 1.000 | 1.000 | 1.000 |
| 0 dB | 0.961 | 0.961 | 0.961 |
| -5 dB | 1.000 | 1.000 | 1.000 |
