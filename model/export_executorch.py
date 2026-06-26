"""Export StressNet to an ExecuTorch ``.pte`` (XNNPACK backend).

    python -m model.export_executorch --output assets/stress_model.pte

Flow:  nn.Module --torch.export--> ExportedProgram
                 --to_edge_transform_and_lower(XnnpackPartitioner)--> Edge
                 --to_executorch--> .pte buffer

XNNPACK (CPU) is the get-it-working-everywhere backend and the demo safety net.
The QNN/Hexagon-NPU lowering (the 40% evidence) is produced through Qualcomm AI
Hub — see model/aihub_profile.py — and on-device with the QNN partitioner.
The ``.pte`` is the contract: no Python, no training deps leak into it.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import executorch
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

from .model import StressNet, build_model, example_input


def _ensure_flatc() -> None:
    """Point ExecuTorch at the flatbuffer compiler it bundles.

    XNNPACK delegate serialization shells out to ``flatc``. The wheel ships one
    at ``executorch/data/bin/flatc``, but it's only on PATH when the venv is
    activated — which it isn't under the Stop/test-gate hook. Set the env var
    ExecuTorch checks (FLATC_EXECUTABLE) so export works regardless of PATH.
    """
    if os.getenv("FLATC_EXECUTABLE") or shutil.which("flatc"):
        return
    for base in executorch.__path__:
        candidate = Path(base) / "data" / "bin" / "flatc"
        if candidate.is_file():
            os.environ["FLATC_EXECUTABLE"] = str(candidate)
            return


def export_to_pte(
    weights: str | None = None,
    *,
    delegate: bool = True,
    model: StressNet | None = None,
) -> bytes:
    """Return the serialized ``.pte`` bytes for StressNet.

    Pass ``model`` to export a specific instance (e.g. the exact net under test,
    so parity is meaningful). Otherwise the model is built from ``weights`` (or
    random init). ``model`` takes precedence over ``weights``.
    """
    _ensure_flatc()
    if model is None:
        model = build_model(weights)
    model.eval()
    example = (example_input(),)

    exported = torch.export.export(model, example)
    partitioners = [XnnpackPartitioner()] if delegate else None
    edge = to_edge_transform_and_lower(exported, partitioner=partitioners)
    et_program = edge.to_executorch()
    return et_program.buffer


def main() -> None:
    ap = argparse.ArgumentParser(description="Export StressNet to a .pte")
    ap.add_argument("--output", "-o", default="assets/stress_model.pte",
                    help="output .pte path")
    ap.add_argument("--weights", "-w", default=None,
                    help="optional checkpoint to load before export")
    ap.add_argument("--no-delegate", action="store_true",
                    help="export portable (no XNNPACK) — for debugging")
    args = ap.parse_args()

    buffer = export_to_pte(args.weights, delegate=not args.no_delegate)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buffer)
    backend = "portable" if args.no_delegate else "XNNPACK"
    print(f"wrote {out}  ({len(buffer):,} bytes, backend={backend})")


if __name__ == "__main__":
    main()
