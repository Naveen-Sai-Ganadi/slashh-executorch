# Combined multi-distortion field robustness (A/B)

Every prior robustness item isolates ONE axis (additive noise, nonlinear clipping, convolutive reverb, linear gain); real capture stacks them at once. This composes them in the physical signal-chain order a mic sees — **gain -> reverb -> noise -> clip** — re-extracts log-mel, and scores. It reports the *compounding gap*: how far simultaneous distortion drops accuracy below the matched single-axis prediction.

- clean (identity) accuracy: **100%**
- worst profile accuracy: **88%**
- graceful across all profiles (bar 80%): **True**
- max compounding gap: **0.069**
- reliable profiles: clean, quiet, typical, harsh, worst_case
- **Graceful composition**: robustness holds under simultaneous distortion — every field profile clears the 80% bar (worst 88%). Single-axis robustness is not overstated, so no combined-distortion augmentation is required. Caveat: worst_case scored ABOVE the matched single-axis baseline (gap -0.156) — stacking more distortion cannot genuinely raise accuracy, so the synthetic eval under-stresses that combination; read the verdict as bounded by synthetic-eval limits, not proof of field robustness.

| profile | gain (dB) | rt60 (s) | snr (dB) | clip_ratio | accuracy | single-axis | compounding gap | reliable |
|---|---|---|---|---|---|---|---|---|
| clean | +0 | 0 | — | 1 | 100% ± 0.000 | — | — | True |
| quiet | -3 | 0.15 | 20 | 0.9 | 100% ± 0.003 | 100% | +0.002 | True |
| typical | -6 | 0.3 | 10 | 0.6 | 99% ± 0.013 | 100% | +0.012 | True |
| harsh | -12 | 0.6 | 0 | 0.35 | 88% ± 0.021 | 95% | +0.069 | True |
| worst_case | -12 | 1 | -5 | 0.25 | 100% ± 0.000 | 84% | -0.156 | True |

_Methodology & caveats: the reverb stage convolves a synthetic exponential-decay RIR whose diffuse tail enters ~20 dB below the direct path (a realistic direct-to-reflection ratio), so the reverb axis is intentionally milder than a fully reverb-dominated room — the harsh and worst_case profiles therefore understate extreme reverberation. A negative compounding gap means a profile scored above its matched single-axis baseline, which is not physically meaningful (stacking more distortion cannot raise accuracy) and flags that the synthetic eval under-stresses that combination rather than demonstrating real robustness._
