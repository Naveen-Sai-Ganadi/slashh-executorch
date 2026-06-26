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


def export_quantized_to_pte(
    model: StressNet,
    calibration: torch.Tensor,
    *,
    per_channel: bool = True,
) -> bytes:
    """Return INT8-quantized ``.pte`` bytes via PT2E + the XNNPACK quantizer.

    Post-training **static** quantization, entirely on the host:
        export --module--> prepare_pt2e --calibrate--> convert_pt2e
              --export--> to_edge_transform_and_lower(XNNPACK) --> .pte

    ``calibration`` is a ``[N, 1, N_MELS, N_FRAMES]`` batch used to observe
    activation ranges. This is the on-host INT8 optimization — it does NOT touch
    Qualcomm AI Hub or the live token (that is the gated M5); it just produces a
    smaller program that still runs through the ExecuTorch runtime.

    ``per_channel`` selects the weight quantization granularity: ``True`` (the
    shipped default) gives each output channel its own scale/zero-point;
    ``False`` uses a single per-tensor scale — smaller and simpler, but coarser.
    """
    # Imported lazily: PT2E quant pulls in torchao, not needed for fp32 export.
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    _ensure_flatc()
    model = model.eval()
    example = (example_input(),)

    captured = torch.export.export(model, example).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel)
    )
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for i in range(calibration.shape[0]):
            prepared(calibration[i:i + 1])
    converted = convert_pt2e(prepared)

    exported = torch.export.export(converted, example)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    return edge.to_executorch().buffer


def main() -> None:
    ap = argparse.ArgumentParser(description="Export StressNet to a .pte")
    ap.add_argument("--output", "-o", default="assets/stress_model.pte",
                    help="output .pte path")
    ap.add_argument("--weights", "-w", default=None,
                    help="optional checkpoint to load before export")
    ap.add_argument("--no-delegate", action="store_true",
                    help="export portable (no XNNPACK) — for debugging")
    ap.add_argument("--quantize", action="store_true",
                    help="INT8 post-training quantization (PT2E + XNNPACK)")
    args = ap.parse_args()

    if args.quantize:
        from .data import synthetic_dataset
        calib, _ = synthetic_dataset(32, seed=1000)
        buffer = export_quantized_to_pte(build_model(args.weights), calib)
        backend = "XNNPACK-INT8"
    else:
        buffer = export_to_pte(args.weights, delegate=not args.no_delegate)
        backend = "portable" if args.no_delegate else "XNNPACK"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buffer)
    print(f"wrote {out}  ({len(buffer):,} bytes, backend={backend})")


if __name__ == "__main__":
    main()
