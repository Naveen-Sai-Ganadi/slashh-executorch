"""End-to-end smoke for the README Quickstart (M11, host half).

The README promises a clean-checkout path: train → export a ``.pte`` → run it
through the ExecuTorch runtime. This test exercises exactly that path through
the public APIs the Quickstart documents, so a regression that breaks the
documented commands fails CI instead of greeting the next human at the prompt.

It is deliberately tiny (a few synthetic epochs) — the point is that the path
*works and stays numerically faithful*, not that the model is accurate. Pure
host: no device, no AI Hub, no live token.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model.export_executorch import export_to_pte
from model.model import example_input
from model.run_pte import run_pte
from model.train import train


def _trained_model():
    model, meta = train(
        data_dir=None, epochs=3, batch_size=16, lr=1e-3, n_per_class=32, seed=0,
    )
    return model, meta


def test_quickstart_trains_exports_and_runs(tmp_path: Path) -> None:
    """train → export .pte → run_pte yields a valid stress score in [0, 1]."""
    model, meta = _trained_model()
    assert meta["source"] == "synthetic"
    assert 0.0 <= meta["val_acc"] <= 1.0

    pte = tmp_path / "stress_model.pte"
    pte.write_bytes(export_to_pte(model=model))
    assert pte.stat().st_size > 0

    out = run_pte(pte, example_input())
    score = out.flatten().tolist()
    assert len(score) == 1
    assert 0.0 <= score[0] <= 1.0


def test_quickstart_pte_matches_eager(tmp_path: Path) -> None:
    """The exported .pte stays faithful to eager PyTorch within 1e-4 (README claim)."""
    model, _ = _trained_model()
    model.eval()

    x = example_input()
    with torch.no_grad():
        eager = model(x)

    pte = tmp_path / "stress_model.pte"
    pte.write_bytes(export_to_pte(model=model))
    runtime_out = run_pte(pte, x)

    max_abs = (eager - runtime_out).abs().max().item()
    assert max_abs < 1e-4, f"pte/eager drift {max_abs:.2e} exceeds 1e-4"
