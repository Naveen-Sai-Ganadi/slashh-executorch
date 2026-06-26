# Score calibration — are the sigmoid outputs real probabilities?

Over 960 windows (noise sweep): scores are **under-confident** (ECE 0.242, signed gap -0.242) — the raw 0.6 gate is biased; temperature scaling (T=0.41) would cut ECE to 0.102.

- **ECE** 0.2423  ·  **MCE** 0.3874  ·  **Brier** 0.0754
- classification accuracy 0.993
- temperature scaling: **T=0.410** → ECE 0.1017 (from 0.2423)
- empirical stress rate near the 0.6 gate (±0.1): **0.926**

Reliability table (confidence in the predicted class vs realised accuracy per bin):

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| 0.5–0.6 | 158 | 0.568 | 0.956 | -0.387 |
| 0.6–0.7 | 214 | 0.656 | 1.000 | -0.344 |
| 0.7–0.8 | 95 | 0.765 | 1.000 | -0.235 |
| 0.8–0.9 | 491 | 0.847 | 1.000 | -0.153 |
| 0.9–1.0 | 2 | 0.901 | 1.000 | -0.099 |

_Calibration is measured across the noise sweep, where scores spread; the clean set alone pins scores near 0/1 and is trivially calibrated. A positive gap = over-confident (the score claims more certainty than it earns)._
