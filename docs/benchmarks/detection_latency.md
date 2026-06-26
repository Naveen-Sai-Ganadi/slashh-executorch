# Detector onset/offset latency (A/B)

The EMA + dual-threshold hysteresis that keeps the latch from flickering also delays it. This drives the real detector through calm→stressed→calm episodes and measures **time-to-alarm**: with `HOP_SECONDS`=1s each window is a second of wall-clock.

- detection rate: **100%** (20 episodes)
- median onset latency: **1s** (worst 3s, budget 3s)
- release: **100%** released, median **2s** after offset
- responsive: **True**
- **Responsive**: the detector fires on every episode with a median onset latency of 1s (worst 3s, budget 3s) and releases a median 2s after stress ends. The EMA + hysteresis smoothing buys flicker-freedom without making the alarm feel sluggish.

| episode | onset (win) | onset (s) | released | release (s) |
|---|---|---|---|---|
| 0 | 1 | 1s | True | 1s |
| 1 | 1 | 1s | True | 1s |
| 2 | 1 | 1s | True | 1s |
| 3 | 1 | 1s | True | 1s |
| 4 | 1 | 1s | True | 1s |
| 5 | 1 | 1s | True | 1s |
| 6 | 1 | 1s | True | 1s |
| 7 | 1 | 1s | True | 1s |
| 8 | 1 | 1s | True | 2s |
| 9 | 1 | 1s | True | 2s |
| 10 | 1 | 1s | True | 2s |
| 11 | 1 | 1s | True | 3s |
| 12 | 1 | 1s | True | 5s |
| 13 | 1 | 1s | True | 5s |
| 14 | 1 | 1s | True | 4s |
| 15 | 1 | 1s | True | 10s |
| 16 | 2 | 2s | True | 3s |
| 17 | 3 | 3s | True | 3s |
| 18 | 2 | 2s | True | 2s |
| 19 | 2 | 2s | True | 3s |
