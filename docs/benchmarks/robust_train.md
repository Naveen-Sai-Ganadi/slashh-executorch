# Noise-augmented training vs clean training

Same architecture, epochs, and optimizer — only the training data differs. Accuracy is on noise-injected eval sets at each SNR. The **operating floor** is the first SNR where accuracy < 0.80.

- baseline (clean-trained) floor: **10 dB**
- augmented floor: **none (holds)**

| SNR | baseline acc | augmented acc | Δ |
|---|---|---|---|
| clean | 1.000 | 1.000 | +0.000 |
| 20 dB | 0.891 | 1.000 | +0.109 |
| 10 dB | 0.500 | 1.000 | +0.500 |
| 5 dB | 0.500 | 1.000 | +0.500 |
| 0 dB | 0.500 | 1.000 | +0.500 |
