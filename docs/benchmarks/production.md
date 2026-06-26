# Production model — smallest robust StressNet

Architecture **`(4, 8, 16)`** (1,549 params), noise-augmented training, exported to a **12.9 KB** `.pte`.

- clean val accuracy: **0.947**
- operating floor: **-5 dB**

| SNR | accuracy | f1 |
|---|---|---|
| clean | 1.000 | 1.000 |
| 20 dB | 1.000 | 1.000 |
| 10 dB | 0.953 | 0.955 |
| 0 dB | 0.823 | 0.785 |
| -5 dB | 0.500 | 0.000 |
