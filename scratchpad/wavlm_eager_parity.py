"""Host eager-PyTorch reference for the WavLM teacher, fed the BYTE-IDENTICAL
input the Hexagon NPU consumed (scratchpad/wavlm_input.raw, [1,48000] f32).

Reconstructs the merged DeployTeacher (LoRA folded into wavlm-large + stress
head) exactly as export_teacher.py does, runs the same raw waveform through it
in fp32 on the CPU, and prints the logit + sigmoid. This is the numerical-parity
ground truth for the on-NPU logit (-2.46484375, prob 0.0784) produced by
qnn-net-run on the extracted forward_2.bin context binary.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/wavlm_eager_parity.py
"""
import numpy as np
import torch

# train_lora.py top-level does `from model.data import _load_wave`, but data.py
# no longer exports it (stale import). _load_and_merge never calls the wave loader,
# so inject the name purely to let the import resolve — the real merge path is
# untouched.
import model.data as _md
if not hasattr(_md, "_load_wave"):
    def _load_wave(*_a, **_k):  # never invoked on the merge path
        raise RuntimeError("stub _load_wave should not be called")
    _md._load_wave = _load_wave

from model.export_teacher import _load_and_merge

NPU_LOGIT = -2.46484375  # faithful run: forward_0/1 buckets fed to forward_2 on HTP

# byte-exact input the NPU saw
wave = torch.from_numpy(
    np.fromfile("scratchpad/wavlm_input.raw", dtype=np.float32).reshape(1, 48000).copy()
)
print(f"[input] wavlm_input.raw  shape={tuple(wave.shape)}  "
      f"mean={wave.mean():.6f} std={wave.std():.6f} "
      f"min={wave.min():.4f} max={wave.max():.4f}", flush=True)

_eager, deploy, tag = _load_and_merge("wavlm", "broad")
deploy.eval()
print(f"[model] merged DeployTeacher tag={tag}", flush=True)

with torch.no_grad():
    logit = deploy(wave).flatten()[0].item()   # DeployTeacher normalizes internally
prob = 1.0 / (1.0 + np.exp(-logit))

print("=" * 64, flush=True)
print(f"EAGER fp32 (host CPU)  logit = {logit:.8f}   sigmoid = {prob:.6f}", flush=True)
print(f"NPU    (Hexagon V79)   logit = {NPU_LOGIT:.8f}   sigmoid = "
      f"{1.0/(1.0+np.exp(-NPU_LOGIT)):.6f}", flush=True)
d = abs(logit - NPU_LOGIT)
print(f"|Δlogit| = {d:.6f}   decision-agree = {(logit > 0) == (NPU_LOGIT > 0)}", flush=True)
print("=" * 64, flush=True)
