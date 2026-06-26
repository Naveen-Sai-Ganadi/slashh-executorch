"""Tests for the trained architecture A/B experiment (model/ab_experiment.py).

Unlike the raw benchmark harness (random-init, accuracy = chance), this trains
each StressNet width variant briefly on the synthetic task, so the accuracy
numbers are real and the size/latency vs accuracy trade-off is decision-grade.
Picks a recommended variant. Host-only; no AI Hub / live token.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.ab_experiment import ABVariantResult, run_ab
from model.model import StressNet
from model.train import train


def test_train_accepts_an_explicit_model_instance() -> None:
    # the A/B needs to train a specific width, not just the default
    net = StressNet(channels=(8, 16, 32))
    trained, meta = train(
        data_dir=None, epochs=2, batch_size=16, lr=1e-3,
        n_per_class=24, seed=0, model=net,
    )
    assert trained is net  # trained in place, same instance
    assert 0.0 <= meta["val_acc"] <= 1.0
    # the small width has fewer params than the base width
    base = StressNet(channels=(16, 32, 64))
    assert sum(p.numel() for p in net.parameters()) < sum(p.numel() for p in base.parameters())


def test_run_ab_trains_and_compares_variants(tmp_path: Path) -> None:
    variants = {"small": (8, 16, 32), "base": (16, 32, 64)}
    out = run_ab(variants, epochs=3, n_per_class=32, seed=0, out_dir=tmp_path)

    assert {r.name for r in out.results} == {"small", "base"}
    for r in out.results:
        assert isinstance(r, ABVariantResult)
        assert 0.0 <= r.val_acc <= 1.0
        assert r.pte_bytes > 1000
        assert r.eager_latency_ms > 0
        assert r.params > 0

    # a recommendation is made and names one of the variants
    assert out.recommended in variants

    # artifacts written
    assert (tmp_path / "ab_experiment.json").is_file()
    assert (tmp_path / "ab_experiment.md").is_file()
    payload = json.loads((tmp_path / "ab_experiment.json").read_text())
    assert payload["recommended"] in variants
    assert len(payload["results"]) == 2


def test_training_actually_learns_the_synthetic_task() -> None:
    # sanity: with enough epochs the synthetic task is separable above chance
    out = run_ab({"base": (16, 32, 64)}, epochs=12, n_per_class=64, seed=0, out_dir=None)
    assert out.results[0].val_acc >= 0.6
