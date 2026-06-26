"""Tests for the INT8-vs-fp32 robustness comparison (model/int8_robustness.py).

We ship the INT8 `.pte` as the deployable artifact, but parity was only ever
measured on *clean* audio. The real question for an on-device stress meter is
whether INT8 quantization preserves the noise-robustness floor the project
advertises. This harness runs the same noise sweep through both the eager fp32
model and the INT8 runtime and reports the per-SNR gap. Host-only; no device,
no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.int8_robustness import (
    Int8RobustnessResult,
    PteModule,
    int8_robustness,
)
from model.production import build_production_model, quantize_production
from model.run_pte import run_pte
from model.train import train


def _trained_int8(seed: int = 0):
    net = build_production_model()
    train(data_dir=None, epochs=10, batch_size=16, lr=1e-3, n_per_class=48,
          seed=seed, model=net)
    net.eval()
    pte = quantize_production(net, calib_n=12)
    return net, pte


def test_pte_module_matches_run_pte_per_sample() -> None:
    # The adapter must produce exactly what calling run_pte sample-by-sample does,
    # so feeding it to robustness_curve measures the real deployable.
    net, pte = _trained_int8()
    mod = PteModule(pte)
    assert mod.eval() is mod
    x = torch.randn(3, 1, 64, 301)
    batched = mod(x)
    assert tuple(batched.shape) == (3, 1)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.pte"
        p.write_bytes(pte)
        for i in range(3):
            one = run_pte(p, x[i : i + 1]).flatten()[0]
            assert abs(float(batched[i, 0]) - float(one)) < 1e-5


def test_int8_robustness_compares_both_curves(tmp_path: Path) -> None:
    net, pte = _trained_int8()
    out = int8_robustness(
        net, pte, snr_levels=[None, 10.0, 0.0], n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    assert isinstance(out, Int8RobustnessResult)

    # both curves cover the same SNR grid
    fp_snrs = [p.snr_db for p in out.fp32.points]
    q_snrs = [p.snr_db for p in out.int8.points]
    assert fp_snrs == q_snrs == [None, 10.0, 0.0]

    # paired on identical inputs -> INT8 tracks fp32 closely on the clean point
    fp_clean = {p.snr_db: p.accuracy for p in out.fp32.points}[None]
    q_clean = {p.snr_db: p.accuracy for p in out.int8.points}[None]
    assert q_clean >= 0.8
    assert abs(q_clean - fp_clean) <= 0.1

    # per-SNR accuracy deltas are recorded and bounded
    assert len(out.acc_deltas) == 3
    for snr, d in out.acc_deltas:
        assert -1.0 <= d <= 1.0

    # artifacts written and self-consistent
    assert (tmp_path / "int8_robustness.json").is_file()
    assert (tmp_path / "int8_robustness.md").is_file()
    payload = json.loads((tmp_path / "int8_robustness.json").read_text())
    assert "fp32" in payload and "int8" in payload
    assert payload["int8_reliable_floor_db"] == out.int8.reliable_floor_db


def test_int8_robustness_markdown_reports_the_verdict(tmp_path: Path) -> None:
    net, pte = _trained_int8()
    out = int8_robustness(
        net, pte, snr_levels=[None, 10.0], n_per_class=24, seed=1, out_dir=tmp_path,
    )
    md = (tmp_path / "int8_robustness.md").read_text()
    assert "INT8" in md and "fp32" in md
    # a one-line verdict on whether INT8 preserves the floor
    assert "preserves" in md.lower() or "degrades" in md.lower()
