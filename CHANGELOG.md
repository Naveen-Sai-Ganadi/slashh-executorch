# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
All work to date is host-side (XNNPACK CPU, ahead-of-time export); on-device
NPU numbers are gated on a device + live AI Hub run and have not been measured.

## [Unreleased]

### Added

- Host benchmark + A/B harness: per-variant latency, `.pte` size, and eager↔`.pte` parity, with artifacts under `docs/benchmarks/`. `e833db0`
- Host INT8 export path (PT2E + XNNPACK): on the base-width net the `.pte` shrinks ~2.9x (97.9 KB → 33.8 KB), recorded as an A/B row in `docs/benchmarks/`. `82ece14`
- Trained width A/B experiment measuring real validation accuracy against size; recommends the smallest `tiny` `(4, 8, 16)` width (~1,549 params, ~12.9 KB `.pte`), ~7.5x smaller than base at no accuracy loss. `96f0128`
- Host stress-detector oracle in Python — a golden spec of the on-device decision loop (EMA + dual-threshold hysteresis), testable off-device and usable for tuning. `dcbab7e`
- Detector tuning A/B harness sweeping threshold/EMA knobs (accuracy vs. flicker); recommends stress=0.55, release=0.35, alpha=1.0 with artifacts in `docs/benchmarks/detector_tuning.*`. `5e57faf`
- Noise-robustness sweep (accuracy vs. SNR) with an operating-floor report; the clean-trained model is accurate at high SNR but falls toward chance by 10 dB (floor 20 dB). `2fa5b7a`
- Auto-generated benchmarks index (`docs/benchmarks/README.md`) that rolls every benchmark JSON artifact into one legible page, defensive against missing/malformed records. `34b8d61`
- End-to-end Quickstart regression test certifying the documented train → export → run path on a clean checkout, with `.pte`↔eager parity to 1e-4. `78dfaed`
- Production recipe (`model/production.py`): builds the smallest robust StressNet `(4, 8, 16)` with noise-augmented training, exports a ~12.9 KB `.pte`, and records the evidence in `docs/benchmarks/production.{json,md}`. Non-destructive — only writes a `.pte` when `--out` is given. `b8cb16d`
- Host pipeline `HostStressPipeline` (`model/pipeline.py`): waveform → features → model → detector end-to-end, a host mirror of the Android `StressPipeline.kt`, with an integration test over calm/stressed traces plus unvoiced gating and reset. `b9fe9ad`
- CI workflow running the host pytest suite on every push and PR (uv, Python 3.11, verified pins), plus a guard test that parses the workflow's YAML invariants. Fully offline — no live token. `1c08b03`
- INT8 export of the shipped production recipe (`quantize_production`, PT2E + XNNPACK): a deployable ~12.3 KB INT8 `.pte`. At ~1,549 params the program is overhead-dominated, so INT8 only trims the `.pte` ~5% (versus the ~2.9x seen on the base width); its real win is integer NPU compute with unchanged predictions (scores within ~0.004 of eager). `378a816`
- Real-time guarantee: `realtime_factor()` reports median/max processing time and real-time factor for the waveform → state chain, and `tests/test_latency.py` certifies the README real-time claim against hot-path regressions. `df3b999`
- Reproducibility guarantee: `tests/test_reproducibility.py` asserts two seeded `train_production(seed=0)` runs yield identical validation accuracy, bit-identical weights, and a byte-identical `.pte` despite global-RNG churn between them. `2cb46cc`
- Multi-seed robustness evidence: `robustness_curve(..., eval_seeds=...)` averages accuracy/F1 across several eval seeds and records the per-SNR accuracy **std** (`acc_std`). `build_production` now ships the production record averaged over 5 eval seeds, so the advertised floors carry a spread (e.g. 0 dB = 0.842 ± 0.015) instead of resting on one lucky draw. `n_eval_seeds=1` keeps the fast single-seed path.
- INT8↔fp32 robustness comparison (`model/int8_robustness.py`): runs the noise sweep through both the eager fp32 model and the shipped INT8 `.pte` (via the ExecuTorch host runtime) on *identical* inputs, and reports a paired per-SNR accuracy gap plus a verdict. The deployable INT8 model **preserves the fp32 floor** — reliable down to 0 dB with a worst-case −0.008 accuracy delta — so "INT8 on the NPU" holds under noise, not just on clean audio. Recorded in `docs/benchmarks/int8_robustness.{json,md}` and rolled into the index.
- Colored-noise robustness (`model/noise_colors.py`): the SNR sweep gained a `noise_color` (`white`/`pink`/`brown`) so robustness can be measured against realistic acoustic spectra, not just flat white noise. White turns out to be the *worst case* — pink and brown (energy concentrated in low frequencies, away from the discriminative mel bands) hold as well or better, with the floor holding to −5 dB for brown — so the advertised white-noise floor is conservative. The default stays `white`, byte-identical to prior records. Recorded in `docs/benchmarks/noise_colors.{json,md}`.
- Cross-initialization robustness envelope (`model/init_envelope.py`): trains several independent inits of the shipped width and runs each through the same noise sweep, then derives the **conservative envelope** — the least-noisy per-init reliable floor, the SNR every init still clears regardless of the training seed. This is the evidence behind the README's "10 dB is the envelope we stand on": across 5 inits, four hold to 0 dB but one only to 10 dB, so the honest cross-init floor is 10 dB even though a typical init does better. Complements the eval-seed averaging (which varies the eval draw with the model fixed) by measuring variance across the *weights*. Recorded in `docs/benchmarks/init_envelope.{json,md}` and rolled into the index.
- Failure-mode breakdown under noise (`model/noise_failure_mode.py`): the same SNR sweep split by error *direction* — precision/recall plus whether the model **misses stress** (false negatives, the detector goes silent) or **false-alarms** (false positives, it cries wolf), naming the dominant failure at the first sub-threshold SNR. The production model degrades by going **silent**: as noise rises it collapses toward predicting "calm" (recall → 0 by −5 dB), it does not cry wolf — which tells the detector gate to favour a lower stress threshold / faster attack in noisy regimes rather than more smoothing. Recorded in `docs/benchmarks/noise_failure_mode.{json,md}` and rolled into the index.

### Changed

- `train()` gained a `dataset=` parameter enabling noise-augmented training, which raises the operating floor so the tiny net stays accurate through 10 dB SNR — exactly where the clean-trained model collapses toward chance. Below 10 dB a net this small is init-sensitive, so 10 dB is the claim we stand on. `c29a359`
- Robustness records now report the SNR the model is reliable *down to* (`reliable_floor`, lowest SNR ≥ threshold) alongside the first-failing SNR, instead of a bare "operating floor" that printed beside its own sub-threshold accuracy row. The production record reads "reliable down to 0 dB / drops below 0.80 at -5 dB," matching its own table. `f6ec035`
