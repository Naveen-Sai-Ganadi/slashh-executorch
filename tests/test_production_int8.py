"""INT8 export of the *production* (tiny, robust) StressNet.

The base-width INT8 path is covered by ``test_quantize.py``; this pins the
production recipe's own quantized variant — the actual deployable artifact. The
honest finding (verified by spike before these assertions): at ~1.5k params the
program is dominated by fixed runtime overhead, so INT8 barely shrinks the
``.pte`` (~1.05x, not the ~3x of the base width). Its value here is integer
compute on the NPU and *unchanged predictions*, which is what we assert:

  * the int8 program runs through the ExecuTorch runtime and stays a valid
    probability, and
  * its scores track the eager model within a tight tolerance (spike: max
    |int8 - eager| ~0.003 across a clean eval set; we assert < 0.05).

Host-only: PT2E + XNNPACK on the host, no Qualcomm AI Hub, no live token.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from model.export_executorch import export_to_pte
from model.model import example_input
from model.production import (
    build_production,
    quantize_production,
    train_production,
)
from model.run_pte import run_pte


@pytest.fixture(scope="module")
def trained():
    """Train the production net once and share it across the int8 tests."""
    torch.manual_seed(0)
    model, meta = train_production(epochs=12, n_per_class=96, seed=0)
    return model.eval(), meta


def test_quantize_production_runs_and_is_valid(trained) -> None:
    model, _ = trained
    buffer = quantize_production(model)

    assert isinstance(buffer, (bytes, bytearray))
    assert len(buffer) > 1000  # a real serialized program

    with tempfile.TemporaryDirectory() as td:
        pte = Path(td) / "prod_int8.pte"
        pte.write_bytes(buffer)
        out = run_pte(pte, example_input())
    assert tuple(out.shape) == (1, 1)
    score = float(out.flatten()[0])
    assert 0.0 <= score <= 1.0


def test_int8_production_tracks_eager(trained) -> None:
    model, _ = trained
    buffer = quantize_production(model)

    # a clean labeled eval set the model was not calibrated on
    from model.data import synthetic_dataset

    xs, _ = synthetic_dataset(16, seed=4242)
    with torch.no_grad():
        eager = model(xs).flatten()

    with tempfile.TemporaryDirectory() as td:
        pte = Path(td) / "prod_int8.pte"
        pte.write_bytes(buffer)
        diffs = []
        for i in range(xs.shape[0]):
            s = float(run_pte(pte, xs[i : i + 1]).flatten()[0])
            assert 0.0 <= s <= 1.0
            diffs.append(abs(s - float(eager[i])))

    # int8 quantization perturbs scores only slightly (spike: ~0.003)
    assert max(diffs) < 0.05
    # and never flips a confident prediction: same side of 0.5 as eager
    for i in range(xs.shape[0]):
        # only check where eager is confident (avoids boundary noise)
        e = float(eager[i])
        if abs(e - 0.5) > 0.1:
            with tempfile.TemporaryDirectory() as td:
                pte = Path(td) / "p.pte"
                pte.write_bytes(buffer)
                s = float(run_pte(pte, xs[i : i + 1]).flatten()[0])
            assert (s > 0.5) == (e > 0.5)


def test_int8_production_not_larger_than_fp32(trained) -> None:
    model, _ = trained
    int8_bytes = len(quantize_production(model))
    fp32_bytes = len(export_to_pte(model=model))
    # INT8 weights never make the program bigger; for this tiny net the gain is
    # marginal (overhead-dominated), so we assert <= rather than a size ratio.
    assert int8_bytes <= fp32_bytes


def test_build_production_records_int8(tmp_path) -> None:
    import json

    result = build_production(
        epochs=12, n_per_class=96, seed=0, out_dir=tmp_path, pte_out=None
    )
    assert result.int8_bytes is not None
    assert result.int8_bytes <= result.pte_bytes
    assert result.int8_max_abs_diff is not None
    assert result.int8_max_abs_diff < 0.05

    record = json.loads((tmp_path / "production.json").read_text())
    assert record["int8_bytes"] == result.int8_bytes
    assert "int8_max_abs_diff" in record
    # the markdown surfaces the int8 line
    assert "INT8" in (tmp_path / "production.md").read_text()
