# Export — stressnet_v2_broad

arch **v2** · mapping **broad** · calib N=256 (real train features) · val N=512

| metric | eager fp32 | fp32 .pte | int8 .pte |
|---|---|---|---|
| val bal-acc | 0.7966 | 0.7966 | 0.7905 |

- max|fp32.pte − eager| = 7.45e-07
- max|int8.pte − eager| = 7.28e-02
- int8 decision agreement vs eager = 0.9961
- sizes: fp32 979.1 KB · int8 259.6 KB

AI Hub ingest artifacts: `assets/stressnet_v2_broad.ts.pt` (TorchScript), `assets/stressnet_v2_broad.onnx` (ONNX).
