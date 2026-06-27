# Reverberation robustness (A/B)

The suite already covers additive noise, nonlinear clipping, and linear gain; this isolates the *convolutive* distortion a real room produces: reverb. Each waveform is convolved with a synthetic exponential-decay RIR (−60 dB at the RT60, direct path at tap 0), then rescaled back to the original RMS (level/energy held fixed, only temporal smearing remains), re-extracted to log-mel, and scored — the most reverberant room the model tolerates before the smeared envelope breaks detection.

- dry (RT60 0.0s) accuracy: **1**
- reliable reverb ceiling (bar 80%): **1s**
- reverb-tolerant across sweep: **True**
- worst: **98%** at **RT60 0.6s**
- **Reverb-tolerant**: detection clears the 80% bar across the entire RT60 sweep (worst 98% at RT60 0.6s). Room reflections and a decay tail from hands-free / across-the-room capture don't break detection, so no dereverberation is required.

| rt60 (s) | reverb tail energy | accuracy | reliable |
|---|---|---|---|
| 0 | 0% | 100% | True |
| 0.15 | 95% | 99% | True |
| 0.3 | 97% | 99% | True |
| 0.6 | 99% | 98% | True |
| 1 | 99% | 100% | True |
