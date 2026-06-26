"""Numerical parity: eager PyTorch == exported .pte through the runtime.

This is the core correctness gate (plan §9). The exported program is only
trustworthy if running it through the ExecuTorch runtime reproduces the eager
model's output within tolerance. A loosened tolerance to force a pass is a bug.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from model.export_executorch import export_to_pte
from model.model import build_model, example_input
from model.run_pte import run_pte

# XNNPACK delegate is fp32 but reorders/fuses ops; this is tight but realistic
# for the exact same weights run through the delegate vs eager.
MAX_ABS_ERR = 1e-4


def _export_fixture(model) -> Path:
    """Export the EXACT instance under test (not a fresh random net)."""
    buf = export_to_pte(delegate=True, model=model)
    path = Path(tempfile.mkdtemp()) / "stress_model.pte"
    path.write_bytes(buf)
    return path


def test_pte_matches_eager_within_tolerance():
    torch.manual_seed(0)
    model = build_model()
    x = example_input()

    with torch.no_grad():
        eager = model(x)

    pte_path = _export_fixture(model)
    runtime_out = run_pte(pte_path, x)

    assert runtime_out.shape == eager.shape, (
        f"shape mismatch: runtime {tuple(runtime_out.shape)} "
        f"vs eager {tuple(eager.shape)}"
    )
    max_err = (runtime_out - eager).abs().max().item()
    assert max_err < MAX_ABS_ERR, (
        f"parity failed: max abs err {max_err:.2e} >= {MAX_ABS_ERR:.0e}"
    )


def test_output_is_valid_probability():
    """Score must be a single value in [0, 1] — the UI/threshold contract."""
    model = build_model()
    pte_path = _export_fixture(model)
    out = run_pte(pte_path, example_input())
    assert out.numel() == 1
    v = out.flatten()[0].item()
    assert 0.0 <= v <= 1.0, f"score {v} outside [0,1]"
