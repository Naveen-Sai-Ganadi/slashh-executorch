# WavLM teacher — INT8 (w8a8) size optimization

LoRA-merged `microsoft/wavlm-large` + stress head (316M params), quantized to an INT8 ExecuTorch `.pte` for the **XNNPACK/CPU** runtime (variant `relpos_excl`: every Linear except the relative-position path). Distinct from the QNN/HTP *NPU* path (FP16-on-HTP), which is proven separately on the S25 Hexagon.

- size: fp32 ≈ **1262 MB** → INT8 **317 MB** (4.0× smaller)
- calibration: 16 train forwards (per-tensor symmetric w8a8)
- eval: 96 held-out val speakers, paired (identical inputs)
- decision agreement INT8↔fp32: **66/96** (69%)
- |Δlogit|: max **10.050**, mean **2.684**
- balanced-set accuracy: fp32 **0.792** vs INT8 **0.562** (Δ -0.229; chance = 0.500)

**Verdict: not viable — documented negative result.** The 4.0× size win is real, but balanced accuracy collapses from 0.792 to **0.562** — only +0.062 above the 0.500 chance line (Δ -0.229 vs fp32). Decision agreement of 66/96 (69%) is near coin-flip, so the binary decision is **not** preserved in any useful sense. Naive per-tensor w8a8 does not work for this 316M-param transformer; recovering accuracy would need per-channel weights (blocked here by the missing portable `dequantize_per_channel` out-variant) or quantization-aware training. The deployable path for this model stays the QNN/HTP **NPU** artifact (FP16-on-HTP), proven on the S25 Hexagon with |Δlogit| ≈ 0.09.

The max |Δlogit| is **large**: per-tensor INT8 on this transformer shifts raw logit magnitudes substantially, so the INT8 scores are not usable as calibrated probabilities — and (see verdict) here the sign of the logit degrades too, not just its magnitude.
