# Recipe-parameterized cross-initialization envelope

For each recipe, 5 independent inits of the shipped width were trained and run through the same noise sweep (same eval draw). The **envelope** is the least-noisy per-init floor — the SNR every init still clears regardless of the training seed. A deeper *single-seed* floor only moves the advertised number if the *envelope* moves with it (threshold 0.80).

- **Deepest cross-init envelope: `very-aggressive` (train SNRs {clean, 20, 10, 5, 0, -5, -10}) reliable down to -5 dB** across 5 inits — this is **deeper** than the next-best envelope (0 dB) — the recipe moves the cross-init envelope, not just one seed's floor, so the advertised floor can follow it down.

| recipe | train SNRs | mean clean acc | cross-init envelope |
|---|---|---|---|
| `aggressive` | {clean, 20, 10, 5, 0, -5} | 1.000 | 0 dB |
| `very-aggressive` ⭐ | {clean, 20, 10, 5, 0, -5, -10} | 1.000 | -5 dB |
| `extreme` | {clean, 20, 10, 5, 0, -5, -10, -15} | 1.000 | -5 dB |
