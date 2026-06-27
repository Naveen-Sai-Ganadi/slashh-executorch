"""Gating test: can the merged WavLM teacher become an ExecuTorch .pte and run?

This is the prerequisite for ANY on-device / emulator test of WavLM. Unlike
StressNet, the teacher has never been through the ExecuTorch (torch.export ->
XNNPACK -> .pte) path — only TorchScript/ONNX for AI Hub. The real unknown is
whether torch.export can trace the WavLM HF encoder at all.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/export_wavlm_pte.py

Exports fp32 (XNNPACK) and runs it through the ExecuTorch runtime on host,
comparing to the eager merged module. Same .pte + same runtime that would run
on device — just on host CPU here. fp32 is ~1.2GB (won't fit the 3GB emulator);
INT8 is a follow-up only if fp32 export succeeds.
"""
import time
from pathlib import Path

import torch

from model.export_teacher import _load_and_merge
from model.export_executorch import _ensure_flatc

ASSETS = Path("assets")


def main():
    t0 = time.time()
    _ensure_flatc()

    # 1) merged WavLM deploy module (raw wave [B,48000] -> logit)
    eager, deploy, tag = _load_and_merge("wavlm", "broad")
    deploy = deploy.eval()
    print(f"[{tag}] merged ({time.time()-t0:.0f}s)", flush=True)

    ex = torch.zeros(1, 48000)

    # 2) torch.export — THE gating step
    try:
        exported = torch.export.export(deploy, (ex,))
        print(f"[{tag}] torch.export OK ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[{tag}] torch.export FAILED: {repr(e)[:500]}", flush=True)
        return

    # 3) lower to XNNPACK + serialize .pte
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    try:
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        buf = edge.to_executorch().buffer
        out = ASSETS / f"teacher_{tag}.pte"
        out.write_bytes(buf)
        print(f"[{tag}] .pte -> {out} ({len(buf)/1e6:.0f} MB, {time.time()-t0:.0f}s)",
              flush=True)
    except Exception as e:
        print(f"[{tag}] lowering/serialize FAILED: {repr(e)[:500]}", flush=True)
        return

    # 4) run through the ExecuTorch runtime on host + parity vs eager
    from model.run_pte import run_pte
    real = torch.randn(1, 48000) * 0.05
    with torch.no_grad():
        ref = deploy(real).flatten()
    got = run_pte(str(out), real).flatten()
    dmax = (ref - got).abs().max().item()
    print(f"[{tag}] ExecuTorch-runtime parity max|Δ|={dmax:.2e} "
          f"(eager={ref.item():.4f} pte={got.item():.4f}) ({time.time()-t0:.0f}s)",
          flush=True)
    print("OK_WAVLM_PTE", flush=True)


if __name__ == "__main__":
    main()
