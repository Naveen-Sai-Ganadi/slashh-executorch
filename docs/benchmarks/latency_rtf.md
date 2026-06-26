# End-to-end latency & Real-Time Factor

Per-window cost of the full **PCM → log-mel → score** chain on this host (XNNPACK-CPU). Window **3 s**, streaming hop **1 s** — every hop a new window must be scored before the next arrives. **all backends clear real-time**; fastest backend: **`pte-fp32`**.

RTF is per-window wall-clock ÷ audio duration; **RTF(hop)** is the one that must stay < 1 for an always-on detector.

| backend | feature ms | inference ms | total ms (mean / p95) | RTF(window) | RTF(hop) | real-time? |
|---|---|---|---|---|---|---|
| `eager` | 0.363 | 0.533 | 0.897 / 1.179 | 0.0002989 | 0.0008967 | ✅ |
| `pte-fp32` | 0.348 | 0.239 | 0.588 / 0.627 | 0.0001959 | 0.0005876 | ✅ |
| `pte-int8` | 0.353 | 0.278 | 0.631 / 0.714 | 0.0002103 | 0.000631 | ✅ |

Feature extraction dominates the budget for a net this small, so the practical optimization lever is the log-mel front-end, not the model. The figures are this host's CPU path; the on-device NPU only widens the head-room.
