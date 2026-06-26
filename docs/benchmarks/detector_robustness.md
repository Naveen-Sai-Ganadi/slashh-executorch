# End-to-end detector robustness under noise

Multi-window traces (8 windows each) are streamed through the EMA + hysteresis detector — a fresh detector per trace — at each SNR. `detect rate` is on stressed traces, `false alarm` on calm ones, `latency` is the median windows-to-latch on detected traces.

- **Detection floor: 10 dB** — the detector still latches stress on ≥80% of stressed traces while false-alarming on ≤20% of calm ones down to this SNR.

| SNR | detect rate | false alarm | latency (windows) |
|---|---|---|---|
| clean | 1.000 | 0.000 | 1 |
| 20 dB | 1.000 | 0.000 | 1 |
| 10 dB | 1.000 | 0.031 | 1 |
| 0 dB | 0.250 | 0.000 | 2 |
| -5 dB | 0.000 | 0.000 | — |
| -10 dB | 0.000 | 0.000 | — |
