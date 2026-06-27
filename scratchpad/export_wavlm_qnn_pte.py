"""Lower the LoRA-merged WavLM DeployTeacher to a QNN/Hexagon-delegated
ExecuTorch .pte for the S25 (Path A — re-lower through ExecuTorch's own QNN
backend so Module.load() can run it, unlike the AI Hub qnn_context_binary .bin).

Runs ONLY on Linux x86_64 inside Docker (the QNN backend ships a compiled native
module; ExecuTorch auto-downloads the QNN SDK there). Use the companion
build_wavlm_qnn_pte.sh to run this from macOS.

Input : raw waveform [1,48000] (in-graph normalize -> WavLM encoder -> mean ->
        head -> [1,1] logit). Host smoke test already proved torch.export is
        bit-exact (max|Δ|=0) at this shape.
Target: Samsung Galaxy S25 / Snapdragon 8 Elite = QcomChipset.SM8750 (HTP V79).
Quant : PT2E + QnnQuantizer w8a8, calibrated on the 32 real normalized waveforms
        in teacher_wavlm_broad_calib.pt; FP16-on-HTP fallback if quant flow dies.

Output: assets/teacher_wavlm_broad_qnn.pte
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())  # repo root (mounted at /work) so `model` imports

import torch

# --- shim the model.data._load_wave regression so model.train_lora imports.
# Not exercised here (calibration comes from the prebuilt .pt) but the
# `from model.data import _load_wave` at train_lora import-time must resolve.
import model.data as _md
if not hasattr(_md, "_load_wave"):
    def _load_wave(wav_path):  # pragma: no cover
        import soundfile as sf
        import torch as _t
        wav, _sr = sf.read(str(wav_path), dtype="float32")
        return _t.from_numpy(wav)
    _md._load_wave = _load_wave

OUT = "assets/teacher_wavlm_broad_qnn.pte"
CALIB = "assets/teacher_wavlm_broad_calib.pt"
N_CALIB = int(os.getenv("N_CALIB", "8"))   # forwards are slow under x86 emulation


def _calibration(n):
    blob = torch.load(CALIB, map_location="cpu", weights_only=False)
    calib = blob["calib"]
    print(f"[calib] {min(n, calib.shape[0])} real normalized waveforms "
          f"(shape {tuple(calib.shape)})", flush=True)
    return calib[:n].contiguous()


def _fold_weight_norm(module):
    """WavLM's positional conv embedding (pos_conv_embed.conv) is wrapped in weight
    normalization, so its weight is produced at runtime by a torch._weight_norm
    node. ExecuTorch's QNN CanonicalizeConv pass rewrites that conv1d->conv2d but
    mis-orders it relative to the _weight_norm node ('used before it has been
    defined'). Folding weight_norm back into a plain `weight` parameter is
    numerically identical and removes the _weight_norm node entirely, so the conv
    canonicalization is clean. Handles both the legacy (weight_g/weight_v) and the
    new parametrize API."""
    import torch.nn.utils as U
    try:
        import torch.nn.utils.parametrize as P
    except Exception:  # noqa: BLE001
        P = None

    removed = 0
    for _name, m in module.named_modules():
        is_legacy = hasattr(m, "weight_g") and hasattr(m, "weight_v")
        is_param = (
            P is not None
            and hasattr(m, "parametrizations")
            and "weight" in getattr(m, "parametrizations", {})
        )
        if is_legacy:
            try:
                U.remove_weight_norm(m)
                removed += 1
                continue
            except Exception:  # noqa: BLE001
                pass
        if is_param:
            try:
                P.remove_parametrizations(m, "weight", leave_parametrized=True)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    print(f"[wn] folded weight_norm in {removed} conv module(s)", flush=True)


class _ExportTeacher(torch.nn.Module):
    """DeployTeacher with a QNN-friendly in-graph normalize.

    DeployTeacher.forward divides by ``wave.std(dim=-1)``, and torch.std lowers to
    a single ``aten.var.correction`` op. ExecuTorch's QNN partitioner has NO node
    visitor for that op, so partitioning aborts hard with
    ``KeyError: 'aten.var.correction'`` (it's not even a graceful CPU fallback).

    We compute the variance manually from primitives (sub / mul / mean / sqrt /
    div / scalar-add) — every one of which the partition log shows QNN supports —
    so the normalize lowers cleanly onto the HTP. This is the population variance
    (divide by N) vs torch.std's unbiased (N-1); for N=48000 that's a ~1e-5
    relative difference in the normalized input, swamped by the encoder and with
    zero effect on the stress decision. The encoder + head are shared by reference
    with the original DeployTeacher, so the merged LoRA weights are unchanged."""

    def __init__(self, encoder: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, wave: torch.Tensor) -> torch.Tensor:  # [B,48000] -> [B,1]
        mean = wave.mean(dim=-1, keepdim=True)
        diff = wave - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)   # population variance
        std = torch.sqrt(var)
        w = diff / (std + 1e-7)
        h = self.encoder(w).last_hidden_state.mean(dim=1)
        return self.head(h)


def _patch_transformers_for_export():
    """transformers' WavLM encoder forward calls is_deepspeed_zero3_enabled() and
    is_fsdp_managed_module(self) (modeling_wavlm.py:485); these reach
    importlib.util.find_spec, which torch.export's dynamo cannot trace -> the
    export aborts. On a single device both are always False, so replace them with
    constant lambdas. Numerically inert (same single-device forward), and dynamo
    now sees a plain constant. Must run AFTER the WavLM module is imported."""
    import importlib
    repl = {
        "is_fsdp_managed_module": lambda *a, **k: False,
        "is_deepspeed_zero3_enabled": lambda *a, **k: False,
    }
    targets = [
        "transformers.models.wavlm.modeling_wavlm",
        "transformers.integrations.fsdp",
        "transformers.integrations.deepspeed",
        "transformers.modeling_utils",
    ]
    patched = []
    for modname in targets:
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        for nm, fn in repl.items():
            if hasattr(mod, nm):
                setattr(mod, nm, fn)
                patched.append(f"{modname}.{nm}")
    print(f"[patch] export-safe transformers guards -> {patched}", flush=True)


def main():
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from model.export_teacher import _load_and_merge

    t0 = time.time()
    print("[1/5] rebuild + merge LoRA into WavLM (from HF cache) ...", flush=True)
    deploy = _load_and_merge("wavlm", "broad")[1].eval()
    # torch.std in DeployTeacher.forward -> aten.var.correction, which QNN's
    # partitioner has no node visitor for (KeyError). Re-wrap with a QNN-friendly
    # manual-variance normalize (shares encoder+head by reference).
    deploy = _ExportTeacher(deploy.encoder, deploy.head).eval()
    _patch_transformers_for_export()  # neutralize untraceable fsdp/deepspeed guards
    _fold_weight_norm(deploy)         # fold pos_conv weight_norm -> plain weight
    example = (_calibration(1),)  # [1,48000]
    print(f"      DeployTeacher ready, example {tuple(example[0].shape)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Capture ONCE with strict=False (dynamo can't trace transformers' control
    # flow even after the guard patch). Both the quant path AND the lowering path
    # consume this already-traced GraphModule, so no raw transformers module is
    # ever re-exported in strict mode inside to_edge_transform_and_lower_to_qnn.
    # That is what makes the FP16 fallback path safe, not just the INT8 path.
    print("[2/5] torch.export capture (strict=False) ...", flush=True)
    captured = torch.export.export(deploy, example, strict=False).module()
    print(f"      captured graph ({time.time()-t0:.0f}s)", flush=True)

    # `captured` is a self-contained GraphModule with its own lifted parameters;
    # the source `deploy` (a ~1.2GB fp32 WavLM-large encoder+head) is no longer
    # referenced by anything downstream. Drop it now so it isn't resident during
    # the memory-heavy QNN lowering + serialization (the 7.65GB Docker VM OOM-
    # killed the prior run *after* QNN finalize, during .to_executorch()).
    import gc
    del deploy
    gc.collect()

    model_to_lower = captured
    use_fp16 = True
    if os.getenv("FP16_ONLY") == "1":
        print("[3/5] FP16_ONLY=1 -> skip INT8, lower captured graph as FP16 on HTP",
              flush=True)
    else:
        try:
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
            from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

            print("[3/5] PT2E QnnQuantizer (w8a8) prepare + calibrate ...", flush=True)
            quantizer = QnnQuantizer()
            prepared = prepare_pt2e(captured, quantizer)
            calib = _calibration(N_CALIB)
            print(f"      calibrate on {calib.shape[0]} samples (slow under emulation) ...",
                  flush=True)
            with torch.no_grad():
                for i in range(calib.shape[0]):
                    prepared(calib[i:i + 1])
                    print(f"      calib {i+1}/{calib.shape[0]} ({time.time()-t0:.0f}s)",
                          flush=True)
            model_to_lower = convert_pt2e(prepared)
            use_fp16 = False
            print(f"      quant OK -> w8a8 ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[quant] quantizer flow unavailable ({e}); FP16 on HTP", flush=True)

    print(f"[4/5] lower to QNN (SM8750 / HTP V79, "
          f"{'fp16' if use_fp16 else 'w8a8'}) ...", flush=True)
    backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8750,            # Snapdragon 8 Elite (S25), HTP V79
        backend_options=backend_options,
    )
    # WavLM's attention builds a *relative position bias* from `torch.arange`
    # (memory_position - context_position). With a fixed [1,48000] input the seq
    # length is constant, so that whole index chain is data-independent — the QNN
    # partitioner groups it into its own subgraph with NO runtime inputs, and QNN
    # cannot serialize an input-less graph ("No graph inputs present for graph[0]"
    # -> err 0x7532 -> empty context binary). Keep `arange` on CPU so its (tiny,
    # run-once) output becomes a real graph input to the delegated ops downstream;
    # the heavy encoder (linear/conv/matmul/layernorm/gelu) still lowers to HTP.
    skip = {"aten.arange.start_step"}
    edge = to_edge_transform_and_lower_to_qnn(
        model_to_lower, example, compiler_specs, skip_node_op_set=skip,
    )
    # QNN has now baked every parameter into the context binary inside `edge`, so
    # the captured fp32 graph (~1.2GB) is dead weight. .to_executorch() needs a
    # transient ~2x copy of the ~630MB program buffer, so reclaim the graph first
    # to keep the serialization peak under the 7.65GB Docker VM ceiling.
    del captured, model_to_lower, example
    gc.collect()

    prog = edge.to_executorch()
    del edge
    gc.collect()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(prog.buffer)
    print(f"[5/5] wrote {OUT}  ({len(prog.buffer):,} bytes, "
          f"{'w8a8' if not use_fp16 else 'fp16'}, SM8750/V79) ({time.time()-t0:.0f}s)", flush=True)

    _extract_qnn_runtime_libs()
    print("WAVLM_QNN_PTE_DONE", flush=True)


def _extract_qnn_runtime_libs():
    """Copy the on-device QNN runtime .so's (arm64 + Hexagon V79 skel) out of the
    auto-downloaded SDK into a staging dir so they can ship in the APK."""
    import glob
    import shutil

    roots = []
    if os.getenv("QNN_SDK_ROOT"):
        roots.append(os.environ["QNN_SDK_ROOT"])
    import executorch.backends.qualcomm as q
    roots += [os.path.join(p, "sdk", "qnn") for p in q.__path__]
    roots += glob.glob(os.path.expanduser("~/.cache/executorch/qnn*"))
    roots += glob.glob(os.path.expanduser("~/.cache/executorch/qnn/**/"), recursive=True)

    dest = "scratchpad/wavlm_qnn_jniLibs/arm64-v8a"
    os.makedirs(dest, exist_ok=True)
    essential = [
        "libQnnHtp.so", "libQnnSystem.so",
        "libQnnHtpV79.so", "libQnnHtpV79Stub.so",
        "libQnnHtpV79Skel.so", "libQnnHtpV79CalculatorStub.so",
    ]
    copied, seen = 0, set()
    for name in essential:
        for root in roots:
            hits = glob.glob(os.path.join(root, "**", name), recursive=True)
            if hits and name not in seen:
                shutil.copy2(hits[0], os.path.join(dest, name))
                seen.add(name); copied += 1
                print(f"[libs] {name}", flush=True)
                break
    missing = [n for n in essential if n not in seen]
    print(f"[libs] copied {copied}/{len(essential)} essential QNN libs -> {dest}", flush=True)
    if missing:
        print("[libs] WARNING missing:", missing, "| roots:", roots[:3], flush=True)


if __name__ == "__main__":
    sys.exit(main())
