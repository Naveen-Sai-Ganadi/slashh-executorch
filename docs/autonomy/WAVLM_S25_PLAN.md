# WavLM → Samsung Galaxy S25 NPU — execution runbook

**Goal:** run the fine-tuned WavLM teacher on the S25 Ultra's Snapdragon NPU (SM8750 /
HTP V79) end-to-end through ExecuTorch's `Module.load` runtime — live mic → raw
waveform `[1,48000]` → on-NPU inference → score — and prove from logcat that it
runs on the Hexagon NPU, not a CPU fallback.

**Device:** R5CXC1XMCDL / SM-S938U (Galaxy S25 Ultra), Android 15, 16 KB pages,
connected + authorized over adb.

## Status of the two paths (decided)

- **Path B — wrap the AI Hub QNN context binary into a .pte — BLOCKED.**
  `assets/teacher_wavlm_broad_npu_int8_qnn.bin` is QAIRT **2.45**; ExecuTorch 1.2.0
  is QAIRT **2.37**. `from_context_binary` fails: *"Using newer context binary on
  old SDK"* / err 5000 (proven `scratchpad/wavlm_ctx_wrap2.log`, 2026-06-27 02:53).
  Do not retry on ET 1.2.0.
- **Path A — re-lower the LoRA-merged WavLM through ExecuTorch's own 2.37 QNN
  backend — CHOSEN.** Produces a 2.37 `.pte` that matches, by construction, the
  2.37 QNN stack `s25-dev` already ships. This is the route.

## What already exists (do not redo)

- AI Hub run: WavLM is **100% on NPU** (1302/1302 ops, 0 CPU) — `jgko06925` quant,
  `j5qz14yn5` compile, `jglo88klg` profile. Required a zero-bias patch on bias-less
  Conv1d (`scratchpad/add_conv_bias.py`). This validates WavLM *can* be fully NPU.
- `s25-dev` worktree: QNN-enabled `executorch-qnn.aar` (16 KB-aligned), 6 V79
  runtime libs in `android/app/src/main/jniLibs/arm64-v8a/`, on-device "calibrate to
  my voice". The committed "NPU artifact #1" .pte is the 83 KB StressNet, **not**
  WavLM.

## Steps

1. **Export the WavLM QNN .pte (Path A).** `bash scratchpad/build_wavlm_qnn_pte.sh`
   (Docker linux/amd64; mounts HF cache + QAIRT 2.37 SDK). Output
   `assets/teacher_wavlm_broad_qnn.pte` + staged libs in
   `scratchpad/wavlm_qnn_jniLibs/`. INT8 (w8a8) preferred; **fall back to
   FP16-on-HTP** (`use_fp16=True`, no calib loop) if INT8 calibration stalls under
   x86 emulation — still fully NPU-resident.
2. **Verify delegation.** Confirm the .pte is QNN-delegated (not all-CPU). Check
   partition/delegate count; compare op coverage against AI Hub's 100%.
3. **Wire the Android raw-waveform WavLM path** in `s25-dev`: a classifier that
   `Module.load`s the WavLM .pte and runs `forward` on `[1,48000]` (mirror the dead
   `WavLmProbe.kt`, but live). Select it in `MainActivity` alongside StressNet.
4. **Push + install.** `adb push assets/teacher_wavlm_broad_qnn.pte
   /sdcard/Android/data/ai.slashh/files/`; build + `adb install` the s25-dev APK.
5. **Run + confirm NPU.** Launch, drive mic, read logcat: HTP/QNN backend init,
   per-op delegation, **no CPU fallback line**. Record latency + mem.

## Guardrails

- All Android work stays on the `s25-dev` branch (non-main; pushes allowed). Never
  stage `android/`, `Vad.kt`, `run_jvm_tests.sh` into host-only `main` commits.
- Never echo/commit the AI Hub or W&B token; rotate after the event.
- `main` pushes need a fresh green sentinel (git-guard) — device artifacts don't go
  to main.

See memory: `wavlm-s25-ondevice-path`, `all-models-on-npu-directive`.
