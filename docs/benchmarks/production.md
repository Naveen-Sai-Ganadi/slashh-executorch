# Production model — smallest robust StressNet

Architecture **`(4, 8, 16)`** (1,549 params), noise-augmented training, exported to a **12.9 KB** `.pte`.

- clean val accuracy: **1.000**
- reliable down to: **-5 dB** (accuracy ≥ 0.80)
- drops below 0.80 at: never (holds at all tested SNRs)
- **INT8** (PT2E + XNNPACK) export: **12.3 KB** (scores within 0.0058 of eager). At ~1,549 params the program is overhead-dominated, so INT8's win here is integer compute on the NPU, not size.

The floors above are measured for *this* trained artifact. The conservative claim we stand on across re-trains is **0 dB** — the cross-init reliable floor the aggressive training recipe holds across 5 independent inits (see `model/recipe_envelope.py` and the project README).

| SNR | accuracy | f1 |
|---|---|---|
| clean | 1.000 ± 0.000 | 1.000 |
| 20 dB | 1.000 ± 0.000 | 1.000 |
| 10 dB | 1.000 ± 0.000 | 1.000 |
| 0 dB | 1.000 ± 0.000 | 1.000 |
| -5 dB | 1.000 ± 0.000 | 1.000 |
