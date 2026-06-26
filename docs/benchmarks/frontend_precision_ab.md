# Front-end float32 vs float64 parity (A/B)

The model runs INT8 on the NPU, but the log-mel front-end runs in floating point on the CPU. The same labelled noisy waveforms go through the front-end in **float32** and **float64**, both feature sets feed the same fp32 model, and each row reports the worst score divergence and how many threshold *decisions* flip — a certification that the cheap single-precision path the device runs is well-conditioned.

- worst |Δ score| (f32 vs f64): **1.19e-07** (tolerance 1e-03)
- decision flips: **0** / 960 windows (flip rate 0.0000)
- precision-robust: **True**
- **Front-end is precision-robust**: float32 matches float64 to within 1.19e-07 with **zero** decision flips across 960 windows. The on-device single-precision log-mel front-end loses nothing to numerics — no need for a higher-precision path.

| SNR | max |Δ score| | flips | n | flip rate |
|---|---|---|---|---|
| clean | 1.19e-07 | 0 | 192 | 0.0000 |
| 20 dB | 1.19e-07 | 0 | 192 | 0.0000 |
| 10 dB | 1.19e-07 | 0 | 192 | 0.0000 |
| 0 dB | 1.19e-07 | 0 | 192 | 0.0000 |
| -5 dB | 5.96e-08 | 0 | 192 | 0.0000 |
