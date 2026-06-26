# Noise-augmentation recipe A/B

Each recipe is a pool of training-time SNRs (per window: draw one, or stay clean). Same architecture, epochs, optimizer, and eval draw — only the training data differs. The winner has the **deepest reliable floor** (lowest SNR still ≥ 0.80 accuracy) while keeping clean accuracy ≥ 90%.

- **Recommended: `aggressive`** (train SNRs {clean, 20, 10, 5, 0, -5}) — reliable floor **-5 dB**, clean accuracy 1.000 (≥ 90%).

| recipe | train SNRs | clean acc | reliable floor |
|---|---|---|---|
| `clean-only` | {clean} | 1.000 | none (clean only) |
| `mild` | {clean, 20, 10} | 1.000 | 10 dB |
| `moderate` | {clean, 20, 10, 5} | 1.000 | 0 dB |
| `aggressive` ⭐ | {clean, 20, 10, 5, 0, -5} | 1.000 | -5 dB |
