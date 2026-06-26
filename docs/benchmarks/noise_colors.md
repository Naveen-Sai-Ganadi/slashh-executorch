# Robustness across noise colors

The same waveform-noise SNR sweep is run for each spectral color. White is flat; **pink** (`1/f`) and **brown** (`1/f²`) put more energy in the low frequencies, closer to real traffic/fan/HVAC noise.

- white reliable down to: **0 dB**
- pink reliable down to: **0 dB**
- brown reliable down to: **-5 dB**
- **The floor holds across colors** — the model stays reliable into noisy SNRs for white, pink, and brown noise alike.

| SNR | white acc | pink acc | brown acc |
|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 |
| 20 dB | 1.000 | 1.000 | 1.000 |
| 10 dB | 0.953 | 1.000 | 1.000 |
| 0 dB | 0.922 | 0.992 | 1.000 |
| -5 dB | 0.500 | 0.500 | 1.000 |
