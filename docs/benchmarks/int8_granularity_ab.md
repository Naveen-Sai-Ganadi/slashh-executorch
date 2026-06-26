# Per-channel vs per-tensor INT8 (A/B)

The same trained production net, quantized two ways: the shipped **per-channel** scheme (a scale/zero-point per output channel) vs **per-tensor** (one scale for the whole tensor — smaller, coarser, and sometimes the only mode a fixed-function NPU supports). Both INT8 programs run through the ExecuTorch host runtime on *identical* noisy inputs, so each row is paired.

- per-channel reliable to: **-5 dB** (accuracy ≥ 0.80)
- per-tensor reliable to: **-5 dB**
- program size: per-channel **12,548** B · per-tensor **9,604** B (**+23.5%**)
- max |Δ accuracy|: **0.000** (material ≥ 0.05: **False**)
- **Per-tensor is the cheaper equivalent**: it holds the same reliable floor (-5.0) with no material accuracy gap (max |Δ| 0.000). For this tiny net the per-channel scale vectors buy nothing — per-tensor is +23.5% on file size (12548->9604 bytes).

| SNR | per-channel acc | per-tensor acc | Δ (PT−PC) |
|---|---|---|---|
| clean | 1.000 | 1.000 | +0.000 |
| 20 dB | 1.000 | 1.000 | +0.000 |
| 10 dB | 1.000 | 1.000 | +0.000 |
| 0 dB | 0.961 | 0.961 | +0.000 |
| -5 dB | 1.000 | 1.000 | +0.000 |
