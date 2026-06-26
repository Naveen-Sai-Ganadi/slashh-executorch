# Cross-initialization robustness envelope

5 independent initializations of the shipped width were trained and run through the same noise sweep (same eval draw). The **envelope** is the least-noisy per-init floor — the SNR every init still clears.

- per-init reliable floor: -10 dB … -5 dB (median -5 dB)
- **Conservative envelope: reliable down to -5 dB across every initialization** (threshold 0.80). This is the floor that holds regardless of the training seed — the honest number to stand on.

| init (seed) | clean acc | reliable down to |
|---|---|---|
| 0 | 1.000 | -10 dB |
| 1 | 1.000 | -5 dB |
| 2 | 1.000 | -10 dB |
| 3 | 1.000 | -5 dB |
| 4 | 1.000 | -5 dB |
