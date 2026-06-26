# StressNet host benchmarks (XNNPACK-CPU baseline)

- host: `macOS-26.5.1-arm64-arm-64bit`
- torch: `2.11.0`  · threads: 8

| variant | channels | dtype | eager latency (ms) | pte latency (ms) | speedup | .pt size (KB) | .pte size (KB) | accuracy | parity max abs err |
|---|---|---|---|---|---|---|---|---|---|
| small | (8, 16, 32) | fp32 | 0.630 | 0.647 | 0.97× | 30.5 | 29.9 | 0.500 | 5.96e-08 |
| base | (16, 32, 64) | fp32 | 0.777 | 0.799 | 0.97× | 99.2 | 97.9 | 0.500 | 5.96e-08 |
| base-int8 | (16, 32, 64) | int8 | 0.729 | 0.650 | 1.12× | 99.2 | 33.8 | 0.500 | 2.30e-04 |

_Latency is mean wall-clock per 1×[1,1,64,301] window. The NPU numbers (AI Hub) are compared against these CPU rows._

_Accuracy reflects the weights benchmarked; variants are random-init by default, so 0.5 = chance. Latency, size, and eager↔pte parity are weight-independent and meaningful as shown._
