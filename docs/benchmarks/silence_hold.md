# Detector silence-hold / gap robustness (A/B)

On an unvoiced window the detector freezes its EMA and holds the latch, so a silence right after a stress episode keeps the alarm latched for the whole quiet stretch. This drives the real detector through calm→stress→**silence gap**→calm episodes and measures how long the latch takes to clear after stress ends — in voiced windows (EMA decay budget) vs wall-clock (with the gap). `HOP_SECONDS`=1s.

- release rate: **100%** (15 episodes)
- median voiced-only clear: **2s** (gap-independent)
- median wall-clock clear: **6s** (worst 18s)
- silence inflation: median **3s**, worst **10s**
- holds through silence: **True**; clears within 8s budget: **False**
- **Holds through silence, exceeds budget**: the latch correctly persists through pauses, but a long silence keeps the alarm latched for up to 18s after stress ends (> 8s budget) — the alarm lingers for the whole quiet stretch. If a pause should clear the meter, add a decay-on-silence or a max-hold timeout to the gated path.

| episode | gap (s) | voiced clear (s) | wall-clock clear (s) | inflation (s) | released |
|---|---|---|---|---|---|
| 0 | 0s | 1s | 1s | 0s | True |
| 1 | 3s | 1s | 4s | 3s | True |
| 2 | 10s | 1s | 11s | 10s | True |
| 3 | 0s | 1s | 1s | 0s | True |
| 4 | 3s | 1s | 4s | 3s | True |
| 5 | 10s | 1s | 11s | 10s | True |
| 6 | 0s | 2s | 2s | 0s | True |
| 7 | 3s | 3s | 6s | 3s | True |
| 8 | 10s | 2s | 12s | 10s | True |
| 9 | 0s | 6s | 6s | 0s | True |
| 10 | 3s | 6s | 9s | 3s | True |
| 11 | 10s | 8s | 18s | 10s | True |
| 12 | 0s | 3s | 3s | 0s | True |
| 13 | 3s | 4s | 7s | 3s | True |
| 14 | 10s | 3s | 13s | 10s | True |
