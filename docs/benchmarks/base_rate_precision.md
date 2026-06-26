# Base-rate (prior-shift) precision (A/B)

Balanced accuracy assumes stress is 50% of windows. In the field a phone hears overwhelmingly calm audio, so even a small per-window false-alarm rate makes most alarms false (the base-rate fallacy). Each row takes the detector's measured **TPR/FPR** at the operating threshold and applies Bayes to report the realized **precision of an alarm** at several stress prevalences, plus the break-even prevalence where precision crosses 50%.

- realistic prevalence swept: **2.0%**
- worst break-even prevalence: **0.00%**
- worst alarm precision @ 2.0%: **100.00%**
- usable at realistic prevalence: **True**
- **Usable under field skew**: at a realistic 2.0% stress prevalence every SNR keeps alarm precision ≥ 50% (worst 100.0%). Break-even prevalence tops out at 0.00%, comfortably below deployment reality — most alarms the user sees are real.

| SNR | TPR | FPR | break-even | P@0.5 | P@0.1 | P@0.02 |
|---|---|---|---|---|---|---|
| clean | 1.000 | 0.000 | 0.00% | 100.00% | 100.00% | 100.00% |
| 20 dB | 1.000 | 0.000 | 0.00% | 100.00% | 100.00% | 100.00% |
| 10 dB | 1.000 | 0.000 | 0.00% | 100.00% | 100.00% | 100.00% |
| 0 dB | 1.000 | 0.000 | 0.00% | 100.00% | 100.00% | 100.00% |
| -5 dB | 1.000 | 0.000 | 0.00% | 100.00% | 100.00% | 100.00% |
