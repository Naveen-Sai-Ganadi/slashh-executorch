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

### End-to-end latency & Real-Time Factor

End-to-end PCM→log-mel→score: fastest **`pte-fp32`** at 0.588 ms/window (hop 1.0s, ~1702× real-time head-room) — **all backends real-time**. _(`latency_rtf.json` · [details](latency_rtf.md))_

### Score calibration (reliability & ECE)

Score calibration: **underconfident** (ECE 0.242); temperature T=0.41 → ECE 0.102 — scores under-state certainty — the gate is conservative. _(`calibration.json` · [details](calibration.md))_

### Cross-init calibration-temperature envelope

Calibration-temperature envelope across 5 inits: T in 0.410…0.410 (median 0.410) vs shipped `DEFAULT_TEMPERATURE`=0.41 — stable default — every initialization's deployable (regularized) temperature lands on the same value and the shipped DEFAULT_TEMPERATURE sits inside that spread, so 0.41 is init-independent, not seed-luck. Caveat: the unconstrained NLL optimum is degenerate (collapses to the search floor) on the near-separable synthetic eval, so the temperature is not finitely identifiable here — re-fit on labeled field audio before trusting any value below 0.41. _(`calibration_envelope.json` · [details](calibration_envelope.md))_

### fp32 -> INT8 calibration drift

fp32->INT8 confidence-calibration drift over 640 windows: ECE fp32 0.2395 -> INT8 0.2405 (drift +0.0010); shipped fp32 temperature transfers to the deployed INT8 model: True. _(`int8_calibration_drift.json` · [details](int8_calibration_drift.md))_

### Per-SNR calibration breakdown

Per-SNR calibration across 5 noise levels: ECE grows with noise: True; the confidence read-out is trustworthy at the -5 dB floor: True. _(`calibration_snr.json` · [details](calibration_snr.md))_

### SNR-aware vs global temperature (A/B)

global vs unconstrained-oracle SNR-aware temperature over 5 noise levels: pooled ECE 0.1017 -> 0.0042 (recovers +0.0974), floor +0.2256 at -5 dB; oracle relies on degenerate sub-floor T: True; worth a deployable SNR estimator: False. _(`snr_aware_temperature_ab.json` · [details](snr_aware_temperature_ab.md))_

### fp32 vs INT8 .pte footprint (A/B)

fp32 vs INT8 .pte footprint for the 1549-param net: fp32 13,188B -> INT8 12,548B (1.05x, +4.9%, 26% of the 4x weight-only ceiling); materially smaller: False. _(`pte_footprint.json` · [details](pte_footprint.md))_

### Per-channel vs per-tensor INT8 (A/B)

INT8 weight granularity A/B (per-channel vs per-tensor): per-channel reliable to -5 dB, per-tensor to -5 dB; per-tensor +23.5% on size, max |Δacc| 0.000; per-channel worth its cost: False. _(`int8_granularity_ab.json` · [details](int8_granularity_ab.md))_

### WavLM teacher INT8 (w8a8) size optimization

WavLM teacher INT8 (w8a8, XNNPACK/CPU): 317.2 MB (3.98× smaller), decision-agree 66/96 vs fp32, balanced acc fp32 0.792 -> INT8 0.562 — **negative result: accuracy collapses to ~chance, not deployable; the deployable path stays the QNN/HTP NPU artifact (FP16-on-HTP)**. _(`wavlm_int8.json` · [details](wavlm_int8.md))_

### Front-end float32 vs float64 parity (A/B)

Front-end float32 vs float64 parity: worst |Δscore| 1.19e-07, 0/960 decisions flip; precision-robust: True. _(`frontend_precision_ab.json` · [details](frontend_precision_ab.md))_

### Base-rate (prior-shift) alarm precision (A/B)

Base-rate precision @ 2.0% prevalence: worst alarm precision 100.0%, break-even 0.00%; usable in field: True. _(`base_rate_precision.json` · [details](base_rate_precision.md))_

### Detector onset/offset latency (time-to-alarm)

Detector time-to-alarm over 20 episodes: detection rate 100%, median onset 1s; responsive: True. _(`detection_latency.json` · [details](detection_latency.md))_

### Detector silence-hold / gap robustness

Detector silence-hold over 15 gap episodes: release rate 100%, worst silence inflation 10s; holds through silence: True. _(`silence_hold.json` · [details](silence_hold.md))_

### Input-gain (level) robustness

Input-gain (level) robustness over 9 levels: reliable band [-24, +36] dB; level-invariant: False. _(`gain_robustness.json` · [details](gain_robustness.md))_

### Clipping / saturation robustness

Clipping / saturation robustness over 6 levels: reliable down to clip_ratio 0.25; clip-tolerant: False. _(`clipping_robustness.json` · [details](clipping_robustness.md))_

### Reverberation robustness

Reverberation robustness over 5 RT60 levels: reliable up to RT60 1s; reverb-tolerant: True. _(`reverb_robustness.json` · [details](reverb_robustness.md))_

### Combined multi-distortion field robustness

Combined multi-distortion field robustness over 5 profiles (gain->reverb->noise->clip): **graceful** — every profile clears the bar (worst 88%); max compounding gap 0.069. _(`combined_distortion.json` · [details](combined_distortion.md))_

### Operating-point / ROC sweep

Operating-point / ROC sweep over 9 thresholds: AUC 1.000, accuracy-optimal threshold 0.6; default 0.5 well placed: True. _(`operating_point.json` · [details](operating_point.md))_

### Front-end throughput (loop vs batched)

Batched log-mel front-end (`extract_batch`) is **3.16×** faster than the per-sample loop building 128 windows (parity holds) — host throughput for dataset/A-B builds; device extractor unchanged. _(`feature_batching.json` · [details](feature_batching.md))_

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
