# Export — stressnet_v1_broad

arch **v1** · mapping **broad** · calib N=256 (real train features) · val N=512

| metric | eager fp32 | fp32 .pte | int8 .pte |
|---|---|---|---|
| val bal-acc | 0.7902 | 0.7902 | 0.7810 |

- max|fp32.pte − eager| = 1.76e-06
- max|int8.pte − eager| = 1.09e-01
- int8 decision agreement vs eager = 0.9570
- sizes: fp32 97.9 KB · int8 33.8 KB

AI Hub ingest artifacts: `assets/stressnet_v1_broad.ts.pt` (TorchScript), `assets/stressnet_v1_broad.onnx` (ONNX).
