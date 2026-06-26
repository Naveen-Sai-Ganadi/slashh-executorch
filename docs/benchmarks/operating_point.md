# Operating-point / ROC sweep (A/B)

The per-window classifier thresholds its probability at a fixed 0.5 before the EMA + hysteresis smoothing. This sweeps that decision threshold, computes TPR / FPR / accuracy / F1 / Youden's J at each, integrates the ROC AUC, and asks whether 0.5 is the right operating point.

- ROC AUC (trapezoidal): **1.000**
- default (0.5) accuracy: **99%**
- best accuracy: **100%** at threshold **0.6** (headroom **+1.2%**)
- Youden-optimal threshold: **0.6** (J=1.00)
- F1-optimal threshold: **0.6** (F1=100%)
- 0.5 well placed (tol 2%): **True**
- **Default 0.5 is well placed**: AUC 1.000; the default per-window threshold yields 99% accuracy, within 1.2% of the best (100% at 0.6). No retune of the raw decision boundary is warranted — tune the hysteresis band instead.

| threshold | TPR | FPR | accuracy | F1 | Youden J |
|---|---|---|---|---|---|
| 0.1 | 100% | 100% | 50% | 67% | +0.00 |
| 0.2 | 100% | 76% | 62% | 72% | +0.24 |
| 0.3 | 100% | 74% | 63% | 73% | +0.26 |
| 0.4 | 100% | 39% | 80% | 84% | +0.61 |
| 0.5 | 100% | 2% | 99% | 99% | +0.98 |
| 0.6 | 100% | 0% | 100% | 100% | +1.00 |
| 0.7 | 77% | 0% | 89% | 87% | +0.77 |
| 0.8 | 58% | 0% | 79% | 73% | +0.58 |
| 0.9 | 0% | 0% | 50% | 0% | +0.00 |
