# Export — stressnet_v2_narrow

arch **v2** · mapping **narrow** · calib N=256 (real train features) · val N=512

| metric | eager fp32 | fp32 .pte | int8 .pte |
|---|---|---|---|
| val bal-acc | 0.8677 | 0.8677 | 0.8662 |

- max|fp32.pte − eager| = 1.73e-06
- max|int8.pte − eager| = 6.01e-02
- int8 decision agreement vs eager = 0.9980
- sizes: fp32 979.1 KB · int8 259.6 KB

AI Hub ingest artifacts: `assets/stressnet_v2_narrow.ts.pt` (TorchScript), `assets/stressnet_v2_narrow.onnx` (ONNX).
