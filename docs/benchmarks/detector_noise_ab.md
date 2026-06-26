# Detector A/B in the noise regime

Each detector config is run over the *same* model and the *same* noisy traces (via the end-to-end `detector_robustness` sweep). The winner pushes the **detection floor** deepest into noise while keeping the worst-case false-alarm rate within budget — testing whether a faster attack hears through the noise the default detector goes silent in.

- **Recommended: `lower-threshold`** (stress=0.5, release=0.4, alpha=0.4) — detection floor -10 dB, worst false alarm 0.062 (≤ 20% budget).

| config | stress | release | alpha | detection floor | worst FA | mean detect | mean latency |
|---|---|---|---|---|---|---|---|
| `default` | 0.6 | 0.45 | 0.4 | -5 dB | 0.000 | 0.839 | 1 |
| `lower-threshold` ⭐ | 0.5 | 0.4 | 0.4 | -10 dB | 0.062 | 1.000 | 1 |
| `fast-attack` | 0.5 | 0.4 | 1.0 | 10 dB | 0.688 | 1.000 | 1 |
| `aggressive` | 0.45 | 0.35 | 1.0 | 20 dB | 0.875 | 1.000 | 1 |
