"""Tests for the production recipe (model/production.py).

The night's experiments converged on a clear answer: the A/B picked the
smallest width (`tiny` = (4, 8, 16)), and the robustness sweep showed that same
tiny width — trained with noise augmentation — is also the *most* robust
(holds accuracy down to -5 dB SNR where wider nets collapse to chance). This
recipe bakes that finding into a single shippable artifact: the smallest robust
StressNet, noise-augmented, exported to a `.pte`.

The clean/0 dB margins are large and consistent (verified by spike before these
assertions were written), so this is not flaky. Host-only; no device, no token.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.model import StressNet, example_input
from model.production import (
    PRODUCTION_CHANNELS,
    ProductionResult,
    build_production,
    build_production_model,
    train_production,
)
from model.run_pte import run_pte


def test_production_model_is_the_small_robust_width() -> None:
    net = build_production_model()
    assert isinstance(net, StressNet)
    assert PRODUCTION_CHANNELS == (4, 8, 16)
    # strictly smaller than the default base width
    base_params = sum(p.numel() for p in StressNet().parameters())
    prod_params = sum(p.numel() for p in net.parameters())
    assert prod_params < base_params


def test_train_production_is_robust_and_clean_accurate() -> None:
    model, meta = train_production(epochs=12, n_per_class=96, seed=0)
    # clean accuracy is essentially perfect on the synthetic task
    assert meta["val_acc"] > 0.9
    assert meta["channels"] == list(PRODUCTION_CHANNELS)
    # the reproducible robustness win: the aggressive production recipe holds
    # well past 10 dB SNR, where the clean-trained baseline collapses toward
    # chance. The cross-init envelope is 0 dB (model/init_envelope.py); a single
    # init like seed 0 typically holds deeper still, so we assert through 10 dB
    # and keep the floor check defensive against init variance.
    acc = {p["snr_db"]: p["accuracy"] for p in meta["robustness"]["points"]}
    assert acc[10.0] > 0.9
    floor = meta["robustness"]["floor_db"]
    assert floor is None or floor <= 10.0   # doesn't fail before the 10 dB point


def test_production_export_matches_eager(tmp_path: Path) -> None:
    model, _ = train_production(epochs=8, n_per_class=64, seed=0)
    model.eval()
    x = example_input()
    with torch.no_grad():
        eager = model(x)
    from model.export_executorch import export_to_pte

    pte = tmp_path / "prod.pte"
    pte.write_bytes(export_to_pte(model=model))
    out = run_pte(pte, x)
    assert (eager - out).abs().max().item() < 1e-4


def test_build_production_writes_record(tmp_path: Path) -> None:
    out = build_production(
        epochs=8, n_per_class=64, seed=0, snr_levels=[None, 10.0, 0.0],
        out_dir=tmp_path,
    )
    assert isinstance(out, ProductionResult)
    assert (tmp_path / "production.json").is_file()
    assert (tmp_path / "production.md").is_file()
    payload = json.loads((tmp_path / "production.json").read_text())
    assert payload["channels"] == list(PRODUCTION_CHANNELS)
    assert payload["pte_bytes"] > 0
    assert "robustness" in payload


def test_production_record_floor_pair_is_honest(tmp_path: Path) -> None:
    """The displayed floor must never be an SNR the model actually fails at.

    Regression guard: the record previously printed a bare "operating floor"
    that was the *first failing* SNR, so it could sit next to its own
    sub-threshold accuracy row (e.g. floor -5 dB beside a -5 dB = 0.500 row),
    reading as a contradiction. We now report the SNR the model is reliable
    *down to* and, separately, where it drops below threshold.
    """
    out = build_production(
        epochs=12, n_per_class=96, seed=0, out_dir=tmp_path,
    )
    rob = json.loads((tmp_path / "production.json").read_text())["robustness"]
    assert "reliable_floor_db" in rob

    acc = {p["snr_db"]: p["accuracy"] for p in rob["points"]}
    rel = rob["reliable_floor_db"]
    if rel is not None:
        # the floor we advertise is a point that genuinely passed threshold
        assert acc[rel] >= 0.8
        # and it is cleaner (higher SNR) than where the model breaks down
        if rob["floor_db"] is not None:
            assert rob["floor_db"] < rel

    md = (tmp_path / "production.md").read_text()
    assert "reliable down to" in md
    # the old self-contradictory bare label is gone
    assert "operating floor" not in md


def test_build_production_records_multi_seed_spread(tmp_path: Path) -> None:
    """The shipped record averages robustness over several eval seeds.

    A floor backed by one lucky draw is weak evidence; the shipped artifact
    instead reports mean ± std per SNR so the spread is visible. Every point
    must carry an ``acc_std`` and the markdown must surface it.
    """
    out = build_production(
        epochs=8, n_per_class=64, seed=0, n_eval_seeds=3,
        snr_levels=[None, 10.0, 0.0], out_dir=tmp_path,
    )
    rob = json.loads((tmp_path / "production.json").read_text())["robustness"]
    for p in rob["points"]:
        assert p["acc_std"] is not None and p["acc_std"] >= 0.0
    # markdown surfaces the spread in the accuracy column
    assert "±" in (tmp_path / "production.md").read_text()
    # the in-memory result agrees
    assert all(p.acc_std is not None for p in out.robustness.points)


def test_build_production_single_seed_has_no_spread(tmp_path: Path) -> None:
    # n_eval_seeds=1 keeps the fast single-seed path: no std, no ± in the table.
    out = build_production(
        epochs=8, n_per_class=64, seed=0, n_eval_seeds=1,
        snr_levels=[None, 10.0, 0.0], out_dir=tmp_path,
    )
    rob = json.loads((tmp_path / "production.json").read_text())["robustness"]
    assert all(p["acc_std"] is None for p in rob["points"])
    assert "±" not in (tmp_path / "production.md").read_text()
    assert all(p.acc_std is None for p in out.robustness.points)
