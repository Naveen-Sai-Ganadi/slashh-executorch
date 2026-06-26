# StressNet host benchmarks (XNNPACK-CPU baseline)

- host: `macOS-26.5.1-arm64-arm-64bit`
- torch: `2.11.0`  · threads: 8

| variant | channels | eager latency (ms) | pte latency (ms) | speedup | .pt size (KB) | .pte size (KB) | accuracy | parity max abs err |
|---|---|---|---|---|---|---|---|---|
| small | (8, 16, 32) | 0.609 | 0.708 | 0.86× | 30.5 | 29.9 | 0.500 | 5.96e-08 |
| base | (16, 32, 64) | 0.779 | 0.793 | 0.98× | 99.2 | 97.9 | 0.500 | 5.96e-08 |

_Latency is mean wall-clock per 1×[1,1,64,301] window. The NPU numbers (AI Hub) are compared against these CPU rows._

_Accuracy reflects the weights benchmarked; variants are random-init by default, so 0.5 = chance. Latency, size, and eager↔pte parity are weight-independent and meaningful as shown._
