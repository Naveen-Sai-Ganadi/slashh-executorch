# Production model — smallest robust StressNet

Architecture **`(4, 8, 16)`** (1,549 params), noise-augmented training, exported to a **12.9 KB** `.pte`.

- clean val accuracy: **0.947**
- reliable down to: **0 dB** (accuracy ≥ 0.80)
- drops below 0.80 at: -5 dB
- **INT8** (PT2E + XNNPACK) export: **12.3 KB** (scores within 0.0040 of eager). At ~1,549 params the program is overhead-dominated, so INT8's win here is integer compute on the NPU, not size.

The floors above are measured for *this* trained artifact. Because a net this small is init-sensitive below 10 dB, the conservative claim we stand on across re-trains is 10 dB (see the project README).

| SNR | accuracy | f1 |
|---|---|---|
| clean | 1.000 | 1.000 |
| 20 dB | 1.000 | 1.000 |
| 10 dB | 0.953 | 0.955 |
| 0 dB | 0.823 | 0.785 |
| -5 dB | 0.500 | 0.000 |
