"""S3 — quantize + export a trained unified StressNet checkpoint.

Drives ONE checkpoint (``model/checkpoints/<tag>.pt`` from train_unified.py)
through the full on-host deployment path and certifies parity:

  load (arch-aware: v1->StressNet, v2->StressNetV2)
    -> fp32 .pte                         (XNNPACK-delegated)
    -> INT8 .pte                         (PT2E static, REAL-feature calibration)
    -> host parity: eager fp32 vs fp32.pte vs int8.pte on held-out val features
    -> TorchScript + ONNX                (AI Hub S4 ingest formats; .pte is NOT)

Calibration uses REAL unified train features (not synthetic) so the PT2E
observers see the true activation ranges the model meets in production.

  python -m model.export_v2 --tag stressnet_v2_broad

Writes assets/<tag>.pte, assets/<tag>_int8.pte, assets/<tag>.onnx,
assets/<tag>.ts.pt and docs/benchmarks/export_<tag>.{json,md}.
Host-only; never touches the live AI Hub token.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import build_model, build_model_v2, example_input
from .export_executorch import export_to_pte, export_quantized_to_pte
from .run_pte import run_pte

UNIFIED = Path("datasets/unified/unified.pt")
ASSETS = Path("assets")
BENCH = Path("docs/benchmarks")


def _load_ckpt(tag: str):
    blob = torch.load(Path("model/checkpoints") / f"{tag}.pt", weights_only=False)
    meta = blob.get("meta", {})
    arch = meta.get("arch", "v2")
    model = build_model_v2() if arch == "v2" else build_model()
    model.load_state_dict(blob["model"])
    model.eval()
    return model, arch, meta


def _calib_and_val(mapping: str, n_calib: int = 256, n_val: int = 512):
    """Real calibration batch (from train split) + a labelled val batch."""
    d = torch.load(UNIFIED, weights_only=False)
    x = d["features"].float()
    y = d["y_narrow" if mapping == "narrow" else "y_broad"].float()
    is_val = torch.tensor([s == "val" for s in d["split"]])
    keep = (y.squeeze(1) >= 0)
    tr = (~is_val) & keep
    va = is_val & keep
    g = torch.Generator().manual_seed(2024)
    xtr = x[tr]
    idx = torch.randperm(xtr.shape[0], generator=g)[:n_calib]
    calib = xtr[idx]
    xva, yva = x[va][:n_val], y[va][:n_val]
    return calib, xva, yva


def _bal_acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    p = (pred.flatten() > 0.5).float()
    yy = y.flatten()
    per = []
    for c in (0.0, 1.0):
        m = (yy == c)
        if m.any():
            per.append((p[m] == yy[m]).float().mean().item())
    return sum(per) / len(per) if per else 0.0


def export(tag: str, mapping: str | None = None):
    model, arch, meta = _load_ckpt(tag)
    if mapping is None:
        mapping = meta.get("mapping", "broad")
    ASSETS.mkdir(exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)

    calib, xva, yva = _calib_and_val(mapping)

    # --- eager fp32 reference -------------------------------------------------
    with torch.no_grad():
        eager = model(xva)

    # --- fp32 .pte ------------------------------------------------------------
    fp32_pte = ASSETS / f"{tag}.pte"
    fp32_pte.write_bytes(export_to_pte(model=model, delegate=True))

    # --- INT8 .pte (real-feature calibration) ---------------------------------
    int8_pte = ASSETS / f"{tag}_int8.pte"
    int8_pte.write_bytes(export_quantized_to_pte(model, calib, per_channel=True))

    # --- host parity ----------------------------------------------------------
    fp32_out = torch.stack([run_pte(fp32_pte, xva[i:i + 1]).flatten()
                            for i in range(xva.shape[0])]).flatten()
    int8_out = torch.stack([run_pte(int8_pte, xva[i:i + 1]).flatten()
                            for i in range(xva.shape[0])]).flatten()
    eager_f = eager.flatten()

    rep = {
        "tag": tag, "arch": arch, "mapping": mapping,
        "n_val": int(xva.shape[0]), "n_calib": int(calib.shape[0]),
        "val_bal_acc_eager": _bal_acc(eager_f, yva),
        "val_bal_acc_fp32_pte": _bal_acc(fp32_out, yva),
        "val_bal_acc_int8_pte": _bal_acc(int8_out, yva),
        "max_abs_fp32_vs_eager": (fp32_out - eager_f).abs().max().item(),
        "max_abs_int8_vs_eager": (int8_out - eager_f).abs().max().item(),
        "agree_int8_vs_eager": ((int8_out > 0.5) == (eager_f > 0.5)).float().mean().item(),
        "fp32_pte_bytes": fp32_pte.stat().st_size,
        "int8_pte_bytes": int8_pte.stat().st_size,
    }

    # --- AI Hub ingest formats (TorchScript + ONNX) ---------------------------
    ex = example_input()
    ts_path = ASSETS / f"{tag}.ts.pt"
    torch.jit.trace(model, ex).save(str(ts_path))
    onnx_path = ASSETS / f"{tag}.onnx"
    # legacy exporter (dynamo=False) — no onnxscript dep, matches aihub_pipeline
    torch.onnx.export(model, ex, str(onnx_path),
                      input_names=["logmel"], output_names=["stress"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"logmel": {0: "batch"}, "stress": {0: "batch"}})
    rep["torchscript"] = str(ts_path)
    rep["onnx"] = str(onnx_path)

    (BENCH / f"export_{tag}.json").write_text(json.dumps(rep, indent=2))
    md = (f"# Export — {tag}\n\n"
          f"arch **{arch}** · mapping **{mapping}** · "
          f"calib N={rep['n_calib']} (real train features) · val N={rep['n_val']}\n\n"
          f"| metric | eager fp32 | fp32 .pte | int8 .pte |\n|---|---|---|---|\n"
          f"| val bal-acc | {rep['val_bal_acc_eager']:.4f} | "
          f"{rep['val_bal_acc_fp32_pte']:.4f} | {rep['val_bal_acc_int8_pte']:.4f} |\n\n"
          f"- max|fp32.pte − eager| = {rep['max_abs_fp32_vs_eager']:.2e}\n"
          f"- max|int8.pte − eager| = {rep['max_abs_int8_vs_eager']:.2e}\n"
          f"- int8 decision agreement vs eager = {rep['agree_int8_vs_eager']:.4f}\n"
          f"- sizes: fp32 {rep['fp32_pte_bytes']/1024:.1f} KB · "
          f"int8 {rep['int8_pte_bytes']/1024:.1f} KB\n\n"
          f"AI Hub ingest artifacts: `{ts_path}` (TorchScript), `{onnx_path}` (ONNX).\n")
    (BENCH / f"export_{tag}.md").write_text(md)
    print(f"[{tag}] eager={rep['val_bal_acc_eager']:.3f} "
          f"int8.pte={rep['val_bal_acc_int8_pte']:.3f} "
          f"agree={rep['agree_int8_vs_eager']:.3f} "
          f"int8={rep['int8_pte_bytes']/1024:.0f}KB")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mapping", choices=["narrow", "broad"], default=None)
    args = ap.parse_args()
    export(args.tag, args.mapping)


if __name__ == "__main__":
    main()
