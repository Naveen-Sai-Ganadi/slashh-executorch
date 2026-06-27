"""Make a LoRA-finetuned teacher NPU-ready: merge adapters, export AI Hub ingest.

Per the all-models-on-NPU directive, the four big teachers (wavlm, hubert,
audeering, emotion2vec) are deployed on the Snapdragon NPU, not kept in the
cloud. LoRA shrinks *training* cost, not the deployed graph — so for device we
**merge the adapters back into the full backbone**, attach the trained stress
head, and emit a single self-contained module that maps a raw 16 kHz waveform
straight to a stress logit.

This module does the host-side, zero-credit prep:

    load base encoder + LoRA adapter (model/checkpoints/lora_<bb>_<map>/)
      -> peft merge_and_unload  (fold adapters into the 317M weights)
      -> DeployTeacher: raw wave [B,48000] -> normalize -> encoder -> mean-pool -> head
      -> TorchScript (.ts.pt) + ONNX (.onnx)        AI Hub ingest formats
      -> real-waveform calibration batch (.pt)       for the AI Hub QUANTIZE job
      -> host fp32 parity: eager LoRAStress vs merged DeployTeacher

The actual INT8 + NPU compile/profile happens on AI Hub's QNN jobs (authorized);
this only produces the artifacts they ingest. The local fp32 parity proves the
merge is lossless before any credit is spent.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 -m model.export_teacher \
        --backbone wavlm --mapping broad

Writes assets/teacher_<bb>_<map>.{ts.pt,onnx}, assets/teacher_<bb>_<map>_calib.pt
and docs/benchmarks/teacher_export_<bb>_<map>.json. Host-only; never touches the
live AI Hub token.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from model.train_lora import (
    HF_BASE, LoRAStress, _norm, _split_idx, load_waves,
)

CKPT = Path("model/checkpoints")
ASSETS = Path("assets")
BENCH = Path("docs/benchmarks")


class DeployTeacher(nn.Module):
    """Self-contained raw-waveform -> stress-logit module (LoRA merged in).

    Normalization lives inside the graph so the deployed model takes a raw
    16 kHz window exactly as captured on device — no host-side feature step."""

    def __init__(self, merged_encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = merged_encoder
        self.head = head

    def forward(self, wave: torch.Tensor) -> torch.Tensor:  # [B,48000] -> [B,1]
        w = (wave - wave.mean(dim=-1, keepdim=True)) / (
            wave.std(dim=-1, keepdim=True) + 1e-7)
        h = self.encoder(w).last_hidden_state.mean(dim=1)
        return self.head(h)


def _load_and_merge(backbone: str, mapping: str):
    """Rebuild LoRAStress, load the trained adapter+head, fold LoRA into the
    backbone, and return (eager LoRAStress, merged DeployTeacher)."""
    tag = f"{backbone}_{mapping}"
    blob = torch.load(CKPT / f"lora_{tag}" / "adapter_head.pt",
                      map_location="cpu", weights_only=False)

    eager = LoRAStress(HF_BASE[backbone]).eval()
    # adapter weights were saved as the "lora"-filtered encoder state_dict
    missing, unexpected = eager.encoder.load_state_dict(blob["lora"], strict=False)
    assert not unexpected, f"unexpected adapter keys: {unexpected[:4]}"
    eager.head.load_state_dict(blob["head"])

    # merge_and_unload folds every LoRA delta into the base Linear weights and
    # returns the plain HF encoder (no peft wrappers, no adapter tensors left).
    merged_encoder = eager.encoder.merge_and_unload()
    deploy = DeployTeacher(merged_encoder, eager.head).eval()
    return eager, deploy, tag


def export(backbone: str, mapping: str = "broad", n_calib: int = 32,
           n_parity: int = 64):
    ASSETS.mkdir(exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    eager, deploy, tag = _load_and_merge(backbone, mapping)
    print(f"[{tag}] merged LoRA into backbone ({time.time()-t0:.0f}s)", flush=True)

    # real-waveform calibration + parity batches from the train/val split
    waves = load_waves()
    tr, va, _y = _split_idx(mapping)
    g = torch.Generator().manual_seed(2024)
    calib_idx = tr[torch.randperm(len(tr), generator=g)[:n_calib]]
    calib = _norm(waves[calib_idx]).contiguous()          # graph re-normalizes,
    parity = waves[va[:n_parity]].contiguous()            # but feed real ranges

    # --- host fp32 parity: eager (peft) vs merged (deploy) --------------------
    with torch.no_grad():
        ref = eager(_norm(parity)).flatten()              # LoRAStress takes normed
        got = deploy(parity).flatten()                    # DeployTeacher norms itself
    max_abs = (ref - got).abs().max().item()
    agree = ((ref > 0) == (got > 0)).float().mean().item()
    print(f"[{tag}] merge parity max|Δ|={max_abs:.2e} decision-agree={agree:.4f}",
          flush=True)

    # --- AI Hub ingest formats ------------------------------------------------
    ex = parity[:1].contiguous()                          # [1,48000] sample input
    ts_path = ASSETS / f"teacher_{tag}.ts.pt"
    rep = {"backbone": backbone, "mapping": mapping, "tag": tag,
           "merge_parity_max_abs": max_abs, "merge_decision_agree": agree,
           "input_shape": list(ex.shape), "n_calib": int(n_calib)}
    try:
        ts = torch.jit.trace(deploy, ex)
        ts.save(str(ts_path))
        rep["torchscript"] = str(ts_path)
        rep["torchscript_bytes"] = ts_path.stat().st_size
        print(f"[{tag}] TorchScript -> {ts_path} "
              f"({rep['torchscript_bytes']/1e6:.0f} MB)", flush=True)
    except Exception as e:  # AI Hub can also ingest ONNX; don't abort on TS fail
        rep["torchscript_error"] = repr(e)[:300]
        print(f"[{tag}] TorchScript FAILED: {repr(e)[:160]}", flush=True)

    onnx_path = ASSETS / f"teacher_{tag}.onnx"
    try:
        torch.onnx.export(
            deploy, ex, str(onnx_path),
            input_names=["wave"], output_names=["stress_logit"],
            opset_version=17, dynamo=False,
            dynamic_axes={"wave": {0: "batch"}, "stress_logit": {0: "batch"}})
        rep["onnx"] = str(onnx_path)
        rep["onnx_bytes"] = onnx_path.stat().st_size
        print(f"[{tag}] ONNX -> {onnx_path} "
              f"({rep['onnx_bytes']/1e6:.0f} MB)", flush=True)
    except Exception as e:
        rep["onnx_error"] = repr(e)[:300]
        print(f"[{tag}] ONNX FAILED: {repr(e)[:160]}", flush=True)

    # calibration set for the AI Hub QUANTIZE job (normalized, model-ready)
    calib_path = ASSETS / f"teacher_{tag}_calib.pt"
    torch.save({"calib": calib, "input_name": "wave"}, calib_path)
    rep["calib_path"] = str(calib_path)
    rep["est_int8_mb"] = round(sum(p.numel() for p in deploy.parameters()) / 1e6, 1)

    (BENCH / f"teacher_export_{tag}.json").write_text(json.dumps(rep, indent=2))
    print(f"[{tag}] done ({time.time()-t0:.0f}s) "
          f"~INT8 {rep['est_int8_mb']} MB · parity {max_abs:.1e}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(HF_BASE))
    ap.add_argument("--mapping", choices=["narrow", "broad"], default="broad")
    ap.add_argument("--n-calib", type=int, default=32)
    args = ap.parse_args()
    export(args.backbone, args.mapping, args.n_calib)


if __name__ == "__main__":
    main()
