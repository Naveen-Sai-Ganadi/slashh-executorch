# End-to-end detector robustness under noise

Multi-window traces (8 windows each) are streamed through the EMA + hysteresis detector — a fresh detector per trace — at each SNR. `detect rate` is on stressed traces, `false alarm` on calm ones, `latency` is the median windows-to-latch on detected traces.

- **Detection floor: -5 dB** — the detector still latches stress on ≥80% of stressed traces while false-alarming on ≤20% of calm ones down to this SNR.

| SNR | detect rate | false alarm | latency (windows) |
|---|---|---|---|
| clean | 1.000 | 0.000 | 1 |
| 20 dB | 1.000 | 0.000 | 1 |
| 10 dB | 1.000 | 0.000 | 1 |
| 0 dB | 1.000 | 0.000 | 1 |
| -5 dB | 1.000 | 0.000 | 1 |
| -10 dB | 0.031 | 0.000 | 1 |
