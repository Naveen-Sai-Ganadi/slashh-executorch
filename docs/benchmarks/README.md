# StressNet benchmarks & experiments

_Auto-generated index of the artifacts in this directory (`python -m model.benchmarks_index`). Each entry links to its full table where available._

### Host benchmarks (XNNPACK-CPU)

Variants: `small` (29.9 KB, 0.647 ms), `base` (97.9 KB, 0.799 ms), `base-int8` (33.8 KB, 0.650 ms). _(`benchmark.json` · [details](benchmark.md))_

### Trained architecture A/B

Trained 3 width variant(s); recommended **`tiny`**. _(`ab_experiment.json` · [details](ab_experiment.md))_

### Detector tuning sweep

Best knobs: stress=0.55, release=0.35, alpha=1.0 (acc 1.0, flicker 7). _(`detector_tuning.json` · [details](detector_tuning.md))_

### Noise robustness

Operating floor: **20 dB**. _(`robustness.json` · [details](robustness.md))_

### INT8 vs fp32 robustness

INT8 vs fp32 under noise: fp32 reliable to -5 dB, INT8 to -5 dB — **INT8 preserves the floor**. _(`int8_robustness.json` · [details](int8_robustness.md))_

### INT8 calibration A/B (clean vs noise-aware)

INT8 calibration A/B (clean vs noise-aware): clean reliable to -5 dB, noise-aware to -5 dB — **no difference** — clean calibration is sufficient; ship **`clean`** calibration. _(`int8_calib_ab.json` · [details](int8_calib_ab.md))_

### Robustness across noise colors

Reliable floor by noise color: white -5 dB, pink -5 dB, brown -5 dB — **holds across colors**. _(`noise_colors.json` · [details](noise_colors.md))_

### Failure mode under noise

First failure under noise leans **false alarms (false positives — detector cries wolf)**. _(`noise_failure_mode.json` · [details](noise_failure_mode.md))_

### Cross-initialization envelope

Across 5 independent inits, the conservative envelope is reliable down to **-5 dB** (holds regardless of training seed). _(`init_envelope.json` · [details](init_envelope.md))_

### End-to-end detector robustness

End-to-end detector (8-window traces): latches stress reliably down to **-5 dB** (detect target met, false alarms in tolerance). _(`detector_robustness.json` · [details](detector_robustness.md))_

### Detector A/B under noise

Detector knobs A/B'd under noise: recommend **`lower-threshold`** (stress=0.5, alpha=0.4) — detection floor -10 dB, worst false alarm 0.0625 (≤ 20%). _(`detector_noise_ab.json` · [details](detector_noise_ab.md))_

### Noise-augmented training

Clean-trained floor 10 dB → augmented floor **none (holds at all tested SNRs)**. _(`robust_train.json` · [details](robust_train.md))_

### Augmentation-recipe A/B

A/B'd 4 augmentation recipe(s): recommend **`aggressive`** — reliable floor -5 dB, clean acc 1.0. _(`augmentation_ab.json` · [details](augmentation_ab.md))_

### Recipe-parameterized cross-init envelope

Recipe envelopes across 5 inits: deepest is **`very-aggressive`** reliable to -5 dB — **moves the envelope**. _(`recipe_envelope.json` · [details](recipe_envelope.md))_

### Production model (shipped recipe)

Shipped width **`(4, 8, 16)`** (1,549 params, 12.9 KB .pte), clean acc 1.000, reliable to -5 dB. INT8 variant 12.3 KB (within 0.0035 of eager). _(`production.json` · [details](production.md))_
