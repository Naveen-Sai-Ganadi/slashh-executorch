# Recipe-parameterized cross-initialization envelope

For each recipe, 4 independent inits of the shipped width were trained and run through the same noise sweep (same eval draw). The **envelope** is the least-noisy per-init floor — the SNR every init still clears regardless of the training seed. A deeper *single-seed* floor only moves the advertised number if the *envelope* moves with it (threshold 0.80).

- **Deepest cross-init envelope: `aggressive` (train SNRs {clean, 20, 10, 5, 0, -5}) reliable down to 0 dB** across 4 inits — this is **deeper** than the next-best envelope (10 dB) — the recipe moves the cross-init envelope, not just one seed's floor, so the advertised floor can follow it down.

| recipe | train SNRs | mean clean acc | cross-init envelope |
|---|---|---|---|
| `production` | {clean, 20, 10, 5} | 1.000 | 10 dB |
| `aggressive` ⭐ | {clean, 20, 10, 5, 0, -5} | 1.000 | 0 dB |
