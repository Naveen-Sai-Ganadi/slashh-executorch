# Clipping / saturation robustness (A/B)

`gain_robustness` swept input level with linear scaling; this isolates the *nonlinear* loud-end failure a real mic/ADC produces: clipping. Each waveform is hard-clipped to `clip_ratio * peak`, then rescaled back to the original peak (level held fixed, only flat-topping remains), re-extracted to log-mel, and scored — the most aggressive clipping the model tolerates before saturation breaks detection.

- unclipped (clip_ratio 1.0) accuracy: **1**
- reliable clip floor (bar 80%): **0.25 (clamp to 25% of peak)**
- clip-tolerant across sweep: **False**
- worst: **50%** at **clip_ratio 0.1**
- **Reliable down to a clip floor**: detection holds above the 80% bar only while clip_ratio stays at or above **0.25** (clamping to 25% of peak); below that, saturation distortion drops accuracy to 50% at clip_ratio 0.1. Loud or close-talking capture that flat-tops harder than this will be missed — add input headroom / anti-clip limiting, or train with clipped augmentation to widen tolerance.

| clip_ratio | clipped samples | accuracy | reliable |
|---|---|---|---|
| 1 | 0% | 100% | True |
| 0.5 | 41% | 100% | True |
| 0.25 | 74% | 84% | True |
| 0.1 | 91% | 50% | False |
| 0.05 | 95% | 50% | False |
| 0.02 | 98% | 50% | False |
