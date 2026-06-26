# StressNet trained architecture A/B

Each width variant is trained briefly on the synthetic task, then exported. Accuracy is **real validation accuracy** (not chance), so the size-vs-accuracy trade-off is decision-grade.

| variant | channels | params | val acc | .pte size (KB) | eager latency (ms) | acc / KB |
|---|---|---|---|---|---|---|
| tiny ⭐ | (4, 8, 16) | 1,549 | 1.000 | 12.9 | 0.533 | 0.0776 |
| small | (8, 16, 32) | 5,977 | 1.000 | 29.9 | 0.610 | 0.0335 |
| base | (16, 32, 64) | 23,473 | 1.000 | 97.9 | 0.758 | 0.0102 |

**Recommended: `tiny`** — best validation accuracy, ties broken toward the smaller program.
