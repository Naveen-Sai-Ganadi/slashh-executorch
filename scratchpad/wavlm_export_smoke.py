"""HOST smoke test for Path A: can the merged WavLM DeployTeacher survive
torch.export at a fixed [1,48000] input? This is the make-or-break for lowering
WavLM to a QNN ExecuTorch .pte. Runs natively on the Mac (no Docker, no QNN) —
torch.export is backend-agnostic, so a pass here means the graph capture that
to_edge_transform_and_lower_to_qnn depends on will work in the x86 container too.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/wavlm_export_smoke.py
"""
import sys
import time

import torch

# --- shim the model.data._load_wave regression so model.train_lora imports.
# We don't call it here (calibration comes from the prebuilt .pt), but the
# `from model.data import _load_wave` at train_lora import-time must resolve.
import model.data as _md
if not hasattr(_md, "_load_wave"):
    def _load_wave(wav_path):  # pragma: no cover - not exercised in smoke test
        import soundfile as sf
        import torch as _t
        wav, _sr = sf.read(str(wav_path), dtype="float32")
        return _t.from_numpy(wav)
    _md._load_wave = _load_wave


def main():
    t0 = time.time()
    from model.export_teacher import _load_and_merge

    print("[1/3] rebuild + merge LoRA into WavLM backbone (from HF cache) ...", flush=True)
    _eager, deploy, tag = _load_and_merge("wavlm", "broad")
    deploy = deploy.eval()
    n_params = sum(p.numel() for p in deploy.parameters())
    print(f"      merged DeployTeacher [{tag}] params={n_params/1e6:.0f}M "
          f"({time.time()-t0:.0f}s)", flush=True)

    blob = torch.load("assets/teacher_wavlm_broad_calib.pt",
                      map_location="cpu", weights_only=False)
    example = blob["calib"][:1].contiguous()   # [1,48000], real range
    print(f"[2/3] example input {tuple(example.shape)}; eager forward ...", flush=True)
    with torch.no_grad():
        ref = deploy(example)
    print(f"      eager out shape={tuple(ref.shape)} val={ref.flatten()[0].item():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[3/3] torch.export.export(deploy, (example,)) ...", flush=True)
    try:
        ep = torch.export.export(deploy, (example,))
        n_nodes = sum(1 for _ in ep.graph.nodes)
        print(f"      EXPORT OK — {n_nodes} graph nodes ({time.time()-t0:.0f}s)", flush=True)
        # parity of exported module vs eager
        with torch.no_grad():
            got = ep.module()(example)
        d = (ref - got).abs().max().item()
        print(f"      exported-vs-eager max|Δ|={d:.2e}", flush=True)
        print("WAVLM_EXPORT_SMOKE_OK", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"WAVLM_EXPORT_SMOKE_FAIL: {type(e).__name__}: {repr(e)[:300]}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
