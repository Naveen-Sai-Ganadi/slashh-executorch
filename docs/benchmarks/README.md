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

### Noise-augmented training

Clean-trained floor 10 dB → augmented floor **none (holds at all tested SNRs)**. _(`robust_train.json` · [details](robust_train.md))_

### Production model (shipped recipe)

Shipped width **`(4, 8, 16)`** (1,549 params, 12.9 KB .pte), clean acc 0.947, floor -5 dB. INT8 variant 12.3 KB (within 0.0040 of eager). _(`production.json` · [details](production.md))_
