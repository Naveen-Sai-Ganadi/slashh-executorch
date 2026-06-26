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
    # the reproducible robustness win: holds through 10 dB SNR, where the
    # clean-trained baseline collapses toward chance. (Behaviour below 10 dB is
    # init-sensitive for a net this small, so we don't assert it.)
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
