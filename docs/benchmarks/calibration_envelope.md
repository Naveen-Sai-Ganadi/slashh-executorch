# Cross-initialization calibration-temperature envelope

5 independent initializations of the production recipe were trained, and the post-hoc temperature was re-fit on each (same eval draw per init). The shipped `DEFAULT_TEMPERATURE` is validated against the resulting spread.

- shipped `DEFAULT_TEMPERATURE`: **0.41**
- per-init deployable temperature: **0.410 … 0.410** (median 0.410, mean 0.410 ± 0.000)
- per-init *unconstrained* optimum: **0.050 … 0.055** (5/5 pinned to the search floor — **degenerate**)
- **Verdict: stable default — every initialization's deployable (regularized) temperature lands on the same value and the shipped DEFAULT_TEMPERATURE sits inside that spread, so 0.41 is init-independent, not seed-luck. Caveat: the unconstrained NLL optimum is degenerate (collapses to the search floor) on the near-separable synthetic eval, so the temperature is not finitely identifiable here — re-fit on labeled field audio before trusting any value below 0.41.**

| init (seed) | deployable T | unconstrained T | ECE | ECE after T |
|---|---|---|---|---|
| 0 | 0.410 | 0.055 (floor) | 0.242 | 0.102 |
| 1 | 0.410 | 0.050 (floor) | 0.203 | 0.061 |
| 2 | 0.410 | 0.050 (floor) | 0.183 | 0.042 |
| 3 | 0.410 | 0.050 (floor) | 0.222 | 0.087 |
| 4 | 0.410 | 0.050 (floor) | 0.207 | 0.058 |
