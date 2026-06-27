# WavLM on the Galaxy S25 Ultra Hexagon NPU — Proof of Execution

**Date:** 2026-06-27
**Device:** Samsung Galaxy S25 Ultra, `SM-S938U` · SoC `SM8750` (Snapdragon 8 Elite) · board `sun` · Android 16
**NPU:** Hexagon V79 HTP (compute DSP / cDSP), QAIRT/QNN 2.37.0.250724
**Model:** `assets/teacher_wavlm_broad_qnn.pte` — LoRA-merged `microsoft/wavlm-large` + stress head, raw waveform `[1,48000]` → stress logit, FP16-on-HTP.

## TL;DR

The fine-tuned WavLM teacher **runs on the physical S25 Ultra's Hexagon V79 NPU** — proven from logcat (the real 9.45 MB `libQnnHtpV79Skel.so` loads onto cDSP domain 3 and the graphs execute), with a **faithful logit of −2.46484375** (prob 0.0784) that matches the host eager fp32 reference (−2.55496168, prob 0.0721) to **|Δlogit| = 0.090, decision-agree = True**.

Two findings, both true:

1. **In-app NPU init is BLOCKED on retail SELinux.** The retail SM-S938U runs SELinux **Enforcing**; the `untrusted_app` domain is denied access to `/dev/fastrpc-cdsp` (labelled `vendor_qdsp_device`). ExecuTorch's in-process QNN delegate therefore cannot load the Hexagon skel from inside the app (skel load error 4000). This is a device **policy** limit on a non-rooted retail handset — not a defect in the model artifact, the AAR, or the hardware.
2. **From the shell context the same model runs end-to-end on the NPU.** The `shell` user (uid 2000) *does* have cDSP access. Running the prebuilt aarch64-android `qnn-net-run` over the raw QNN context binaries extracted from the very same `.pte` loads the real V79 HTP skel onto the cDSP and executes the WavLM transformer encoder on the Hexagon NPU. Same weights, same graph, same SoC — the only thing that changes is the SELinux domain the process runs in.

## How the model was run from the shell

The `.pte` is partitioned into **3 QNN context binaries**, chained by CPU glue ops in the ET runtime. They were recovered from the `.pte` with ExecuTorch's `dump_context_from_pte()` (which calls `PyQnnManagerAdaptor.StripProtocol`, x86-linux-only → run in a `linux/amd64` container) and pushed to `/data/local/tmp/qnntest/`:

| binary | size | role |
|---|---|---|
| `forward_0.bin` | 44 KB | `arange[149]` ×2 → `sub_tensor_1` `[149,149]` int32 (relative positions = mem−ctx) |
| `forward_1.bin` | 48 KB | `abs` → `to_copy_6` `[149,149]` int32 (T5/WavLM log-scaled large-bucket path) |
| `forward_2.bin` | 614 MB | WavLM encoder + mean-pool + stress head → `[1,1]` fp32 logit |

`forward_2` inputs: `input_0_wave` f32 `[1,48000]`, `input_1_aten_sub_tensor_1` i32 `[149,149]`, `input_2_aten_abs_default` i32 `[149,149]`, `input_3_aten_minimum_default` i32 `[149,149]`. The "faithful" run feeds `input_3` the actual `to_copy_6` produced by running `forward_0`+`forward_1` **on the HTP** (the rel-pos `min`-clamp is a no-op at seq_len 149, so `to_copy_6` is the faithful `minimum_default`).

Run command (libs + skel co-located in the run dir; `shell` has cDSP access):

```
cd /data/local/tmp/qnntest
export LD_LIBRARY_PATH=/data/local/tmp/qnntest
export ADSP_LIBRARY_PATH=/data/local/tmp/qnntest
./qnn-net-run --backend libQnnHtp.so --retrieve_context forward_2.bin \
  --input_list inlist2f.txt --output_dir out2f --log_level info
```

## Evidence — logcat proves cDSP/HTP, not CPU fallback

Captured fresh on 2026-06-27 05:18:30 after `adb logcat -c`:

```
W QnnDsp  : Initializing HtpProvider
I qnn-net-run: fastrpc_apps_user_init done with default domain:3            # domain 3 = cDSP
I qnn-net-run: multidsplib_env_init: libcdsprpc.so loaded
I cdsprpcd : Successfully opened file /vendor/dsp/cdsp/fastrpc_shell_unsigned_3
I qnn-net-run: Created user PD on domain 3 ... Unsigned:Y                    # unsigned PD on cDSP
I qnn-net-run: Successfully opened file /data/local/tmp/qnntest/./libQnnHtpV79Skel.so   # read:8391us mmap:577us
I qnn-net-run: remote_handle64_open: opened handle ... for
   file:///libQnnHtpV79Skel.so?qnn_skel_handle_invoke&_modver=1.0&_dom=cdsp on domain 3
   (spawn time 28108 us, load time 41876 us), refs 1                        # <-- THE PROOF
...
Executing Graphs
Finished Executing Graphs
```

- `libQnnHtpV79Skel.so` on device is **9,452,344 bytes** — the real Hexagon V79 HTP skel, not the trivial `libQnnHtpV79CalculatorStub.so`.
- `_dom=cdsp ... on domain 3` = the skel is invoked on the **compute DSP** (Hexagon NPU), via FastRPC.
- `Finished Executing Graphs` = the WavLM encoder graph ran to completion on the NPU.

Benign, non-blocking noise in the same log (does not affect execution):
- `Error 0xd: open_shell failed for domain 3 ... (Permission denied)` then `cdsprpcd: Successfully opened .../fastrpc_shell_unsigned_3` — the app-user shell-path probe is denied, but the `cdsprpcd` daemon loads the unsigned PD shell from the standard cDSP path. Normal.
- `error_code 0x2` on `.../cdsp/./libQnnHtpV79Skel.so` then `error_code 0x0` on `.../qnntest/./libQnnHtpV79Skel.so` — first search-path miss, then success from the run dir. Normal.
- `Error 0x80000414 ... libdspqueue_rpc_skel.so method 3` — benign teardown-time dspqueue capability probe; fires after the graph has already executed.

## Numerical result and parity

| run | logit | sigmoid(prob) |
|---|---|---|
| **NPU (Hexagon V79, faithful rel-pos buckets)** | **−2.46484375** | **0.078360** |
| NPU (zeroed `input_3` control) | −4.5625 | 0.0103 |
| Host eager fp32 (`DeployTeacher`, byte-identical input) | −2.55496168 | 0.072094 |

`|Δlogit|` NPU vs eager fp32 = **0.090118**, **decision-agree = True**. The ~0.09 gap is the expected FP16-on-HTP quantization error for a 317M-param transformer encoder. The zeroed-`input_3` control (−4.56) confirms the graph genuinely consumes its position-bias inputs — the NPU is running the real model, not a degenerate constant.

Output bytes for the faithful run: `00 c0 1d c0` (little-endian f32) = −2.46484375. ✔

## Reproduction artifacts

- Context extraction: `scratchpad/extract_qnn_ctx.py` + `scratchpad/run_extract_qnn_ctx.sh` (Docker linux/amd64).
- Host eager parity: `scratchpad/wavlm_eager_parity.py` (`PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/wavlm_eager_parity.py`).
- On-device rig: `/data/local/tmp/qnntest/` — full V79 QNN lib set, `qnn-net-run`, the 3 `.bin` files, `inlist2f.txt`, `out2f/`.

## What this means for the product

The S25 Ultra Hexagon NPU **can and does run the fine-tuned WavLM teacher**. The only thing standing between this and an in-app on-NPU inference is the **retail-device SELinux app-sandbox policy**, which is outside the model/runtime's control on a non-rooted handset. Options to ship on-NPU in-app, in rough order of effort: (a) a privileged/system-app or vendor allowlist for cDSP access; (b) a rooted/dev device for the on-NPU demo; (c) keep the in-app path on the XNNPACK/CPU ExecuTorch runtime and reserve the NPU path for shell/benchmark proof. The artifact and hardware are proven; the blocker is policy.
