# fp32 -> INT8 confidence-calibration drift

`DEFAULT_TEMPERATURE` and the score-calibration analysis were fit on the fp32 eager model, but the device serves the INT8 `.pte`. This checks whether quantization moves the calibration and whether the shipped fp32 temperature still fits the deployed model.

- windows (noise sweep): **640**
- raw ECE: fp32 **0.2395** -> INT8 **0.2405** (drift **+0.0010**)
- own fitted temperature: fp32 **0.41** · INT8 **0.41**
- shipped T=0.41 applied to INT8 -> ECE **0.0991** (transfers: **True**)
- **Verdict: INT8 quantization raises ECE by 0.001 (fp32 0.239 -> INT8 0.241); the drift is within 0.025 and the shipped fp32 temperature (T=0.41) still cuts INT8 ECE to 0.099, so it transfers to the deployed model — no re-fit needed.**

| model | n | raw ECE | fitted T | ECE after fitted T |
|---|---|---|---|---|
| fp32 (eager) | 640 | 0.2395 | 0.41 | 0.0993 |
| INT8 (.pte) | 640 | 0.2405 | 0.41 | 0.0991 |
