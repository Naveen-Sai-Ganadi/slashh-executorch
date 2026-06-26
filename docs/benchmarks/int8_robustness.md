# INT8 vs fp32 noise robustness

The shipped deployable is the INT8 `.pte`. The same waveform-noise sweep is run through both the eager **fp32** model and the **INT8** ExecuTorch runtime on *identical* inputs, so each row is a paired comparison.

- fp32 reliable down to: **0 dB** (accuracy ≥ 0.80)
- INT8 reliable down to: **0 dB**
- **INT8 preserves the floor** — the deployable is reliable down to at least the same SNR as the fp32 model.

| SNR | fp32 acc | INT8 acc | Δ (INT8−fp32) |
|---|---|---|---|
| clean | 1.000 | 1.000 | +0.000 |
| 20 dB | 1.000 | 1.000 | +0.000 |
| 10 dB | 0.953 | 0.953 | +0.000 |
| 0 dB | 0.922 | 0.914 | -0.008 |
| -5 dB | 0.500 | 0.500 | +0.000 |
