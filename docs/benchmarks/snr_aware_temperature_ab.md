# SNR-aware vs global temperature (A/B)

The per-SNR breakdown showed the single global temperature under-corrects at the noisy floor. This bounds the prize: an *oracle* that knows each window's true SNR applies that SNR's own best temperature. The oracle isn't deployable (the device doesn't know the true SNR) — the gap is the upper bound an on-device SNR estimator could chase.

- global `DEFAULT_TEMPERATURE`=0.41; the oracle fits each SNR's own T on a fine grid that may descend below 0.41 (sub-floor `(floor)` temperatures are degenerate — see calibration_envelope)
- pooled ECE: global **0.1017** -> oracle **0.0042** (recovers **+0.0974**)
- floor reduction: **+0.2256**  ·  oracle relies on degenerate sub-floor T: **True**  ·  worth a deployable SNR estimator: **False**
- **Verdict: the prize exists but isn't safely reachable: an oracle recovers 0.226 ECE at the -5 dB floor (0.226->0.000), but only by sharpening below the 0.41 regularization floor — the same degenerate sub-floor temperature calibration_envelope flags on near-separable synthetic audio. An SNR-aware schedule can't claim this gain without real field audio to prove the sub-floor temperature generalizes; until then ship the single global temperature and treat low-SNR confidence as under-stated.**

| SNR | n | ECE (global T) | ECE (oracle T) | oracle T | recovered |
|---|---|---|---|---|---|
| clean | 192 | 0.0146 | 0.0000 | 0.05 (floor) | +0.0146 |
| 20 dB | 192 | 0.0192 | 0.0000 | 0.05 (floor) | +0.0192 |
| 10 dB | 192 | 0.0819 | 0.0001 | 0.05 (floor) | +0.0818 |
| 0 dB | 192 | 0.1665 | 0.0207 | 0.065 | +0.1459 |
| -5 dB | 192 | 0.2261 | 0.0005 | 0.05 (floor) | +0.2256 |
