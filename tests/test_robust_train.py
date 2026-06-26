"""Tests for noise-augmented training (model/robust_train.py).

The robustness sweep showed the clean-trained model collapses toward chance by
~10 dB SNR. This closes that loop: train on a noise-augmented set and show the
operating floor improves. The improvement on the synthetic task is large and
consistent (verified by spike), so these assertions are not flaky.
Host-only; no device, no AI Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.data import synthetic_dataset
from model.robust_train import (
    RobustTrainResult,
    compare_robustness,
    noise_augmented_dataset,
)
from model.train import train


def test_augmented_dataset_shape_determinism_and_noise() -> None:
    x, y = noise_augmented_dataset(n_per_class=8, seed=0)
    assert tuple(x.shape) == (16, 1, 64, 301)
    assert tuple(y.shape) == (16, 1)
    again = noise_augmented_dataset(n_per_class=8, seed=0)[0]
    assert torch.allclose(x, again)                      # deterministic
    # augmentation actually perturbs vs the clean synthetic set
    clean = synthetic_dataset(8, seed=0)[0]
    assert not torch.allclose(x, clean)


def test_train_accepts_a_prebuilt_dataset() -> None:
    x, y = noise_augmented_dataset(n_per_class=16, seed=0)
    from model.model import StressNet

    net = StressNet()
    trained, meta = train(
        data_dir=None, epochs=3, batch_size=16, lr=1e-3,
        n_per_class=0, seed=0, model=net, dataset=(x, y),
    )
    assert trained is net
    assert meta["source"] == "provided"
    assert 0.0 <= meta["val_acc"] <= 1.0


def test_augmented_training_raises_the_operating_floor(tmp_path: Path) -> None:
    out = compare_robustness(
        epochs=12, n_per_class=96, seed=0, eval_n_per_class=96,
        snr_levels=[None, 20.0, 10.0, 5.0, 0.0], out_dir=tmp_path,
    )
    assert isinstance(out, RobustTrainResult)

    base = {p.snr_db: p.accuracy for p in out.baseline.points}
    aug = {p.snr_db: p.accuracy for p in out.augmented.points}
    # both perfect on clean
    assert base[None] > 0.9 and aug[None] > 0.9
    # the win: at 10 dB the clean-trained model is near chance, augmented is not
    assert aug[10.0] > base[10.0] + 0.2
    # augmented floor is at least as deep (numerically lower, or None) as baseline
    assert _floor_rank(out.augmented.floor_db) >= _floor_rank(out.baseline.floor_db)

    # artifacts
    assert (tmp_path / "robust_train.json").is_file()
    assert (tmp_path / "robust_train.md").is_file()
    payload = json.loads((tmp_path / "robust_train.json").read_text())
    assert "baseline" in payload and "augmented" in payload


def _floor_rank(floor_db) -> float:
    # higher rank = more robust; None (never fails) ranks above any numeric floor
    return float("inf") if floor_db is None else -floor_db
