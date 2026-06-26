# Input-gain (level) robustness (A/B)

The log-mel front-end is `log10(mel + eps)` with no level normalization, so scaling the input waveform by gain `g` shifts every log-mel bin by `2*log10(g)`. This sweeps input gain (dB), re-extracts features from gained waveforms, and measures accuracy — the dynamic range the model tolerates before quiet/loud capture breaks detection. Gain is applied after noise, so SNR is held fixed.

- unity (0 dB) accuracy: **1**
- reliable gain band (bar 80%): **[-24, +36] dB (60 dB wide)**
- level-invariant across sweep: **False**
- worst: **50%** at **-60 dB**
- **Reliable within a band**: detection holds above the 80% bar only for input gain in [-24, +36] dB (a 60 dB window); outside it accuracy falls to 50% at -60 dB. Because the front-end isn't level-normalized, very quiet or loud capture shifts the log-mel map out of the learned range — add input AGC or a per-window level normalization to widen the usable range.

| gain (dB) | gain x | accuracy | reliable |
|---|---|---|---|
| -60 | 0.001 | 50% | False |
| -48 | 0.00398 | 50% | False |
| -36 | 0.0158 | 55% | False |
| -24 | 0.0631 | 93% | True |
| -12 | 0.251 | 100% | True |
| +0 | 1 | 100% | True |
| +12 | 3.98 | 100% | True |
| +24 | 15.8 | 100% | True |
| +36 | 63.1 | 86% | True |
