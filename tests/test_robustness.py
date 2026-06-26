"""Tests for the noise-robustness sweep (model/robustness.py).

The app classifies real-world audio that is rarely clean, so a model's accuracy
under additive acoustic noise matters as much as its clean accuracy. This sweep
injects white noise at graduated SNRs **on the waveform** (before the log-mel
feature extractor, where physical noise actually lives) and reports the
accuracy-vs-SNR curve plus the operating floor. Host-only; no device, no AI Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.model import StressNet
from model.robustness import (
    RobustnessPoint,
    noisy_synthetic_dataset,
    operating_floor,
    reliable_floor,
    robustness_curve,
)
from model.data import synthetic_dataset
from model.train import train


def _trained_model(seed: int = 0) -> StressNet:
    net = StressNet()
    train(data_dir=None, epochs=10, batch_size=16, lr=1e-3,
          n_per_class=48, seed=seed, model=net)
    return net.eval()


def test_clean_dataset_matches_plain_synthetic() -> None:
    # snr_db=None means no noise injected -> identical draw to synthetic_dataset
    x0, y0 = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=None)
    x1, y1 = synthetic_dataset(8, seed=0)
    assert torch.allclose(x0, x1) and torch.equal(y0, y1)


def test_noisy_dataset_shapes_and_determinism() -> None:
    x, y = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=10.0)
    assert tuple(x.shape) == (16, 1, 64, 301)
    assert tuple(y.shape) == (16, 1)
    again = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=10.0)[0]
    assert torch.allclose(x, again)
    # noise actually changed the features vs clean
    clean = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=None)[0]
    assert not torch.allclose(x, clean)


def test_robustness_curve_degrades_with_noise(tmp_path: Path) -> None:
    model = _trained_model(seed=0)
    out = robustness_curve(
        model, snr_levels=[None, 20.0, 0.0, -10.0],
        n_per_class=48, seed=1, out_dir=tmp_path,
    )
    assert len(out.points) == 4
    for p in out.points:
        assert isinstance(p, RobustnessPoint)
        assert 0.0 <= p.accuracy <= 1.0

    by_snr = {p.snr_db: p.accuracy for p in out.points}
    clean = by_snr[None]
    assert clean > 0.7                       # learns the clean task
    # heavy noise must not *improve* accuracy over clean
    assert by_snr[-10.0] <= clean + 0.05

    # artifacts
    assert (tmp_path / "robustness.json").is_file()
    assert (tmp_path / "robustness.md").is_file()
    payload = json.loads((tmp_path / "robustness.json").read_text())
    assert len(payload["points"]) == 4

    # the record carries both floors, and the honesty invariant holds: the
    # reliable floor (if any) names a point that genuinely passed threshold.
    assert "reliable_floor_db" in payload
    rel = out.reliable_floor_db
    if rel is not None:
        passed = {p.snr_db for p in out.points if p.accuracy >= 0.8}
        assert rel in passed
        # the failing floor, if any, is strictly noisier than the reliable one
        if out.floor_db is not None:
            assert out.floor_db < rel


def test_operating_floor_finds_first_failing_snr() -> None:
    pts = [
        RobustnessPoint(snr_db=None, accuracy=0.98, f1=0.98, n=96),
        RobustnessPoint(snr_db=20.0, accuracy=0.95, f1=0.95, n=96),
        RobustnessPoint(snr_db=10.0, accuracy=0.82, f1=0.81, n=96),
        RobustnessPoint(snr_db=0.0, accuracy=0.61, f1=0.60, n=96),
    ]
    # first SNR whose accuracy drops below 0.8, scanning clean -> noisy
    assert operating_floor(pts, threshold=0.8) == 0.0
    # nothing below threshold -> None
    assert operating_floor(pts, threshold=0.5) is None


def test_robustness_curve_averages_over_eval_seeds() -> None:
    # Averaging accuracy over several eval seeds (with std) is stronger evidence
    # than a single draw — it answers "is this floor luck?". The multi-seed
    # curve must equal the mean of the per-seed single curves and carry a std.
    model = _trained_model(seed=0)
    levels = [None, 10.0, 0.0]
    seeds = (1, 2, 3)

    multi = robustness_curve(model, snr_levels=levels, n_per_class=32, eval_seeds=seeds)
    singles = [
        robustness_curve(model, snr_levels=levels, n_per_class=32, seed=s)
        for s in seeds
    ]

    assert len(multi.points) == len(levels)
    for i, p in enumerate(multi.points):
        accs = [s.points[i].accuracy for s in singles]
        assert abs(p.accuracy - sum(accs) / len(accs)) < 1e-3
        assert p.acc_std is not None and p.acc_std >= 0.0

    # deterministic for the same eval seeds
    again = robustness_curve(model, snr_levels=levels, n_per_class=32, eval_seeds=seeds)
    assert [p.accuracy for p in multi.points] == [p.accuracy for p in again.points]
    assert [p.acc_std for p in multi.points] == [p.acc_std for p in again.points]

    # single-seed path is unchanged: no std reported
    one = robustness_curve(model, snr_levels=levels, n_per_class=32, seed=1)
    assert all(p.acc_std is None for p in one.points)


def test_reliable_floor_is_lowest_passing_snr() -> None:
    pts = [
        RobustnessPoint(snr_db=None, accuracy=0.98, f1=0.98, n=96),
        RobustnessPoint(snr_db=20.0, accuracy=0.95, f1=0.95, n=96),
        RobustnessPoint(snr_db=10.0, accuracy=0.82, f1=0.81, n=96),
        RobustnessPoint(snr_db=0.0, accuracy=0.61, f1=0.60, n=96),
    ]
    # lowest numeric SNR still >= threshold, scanning clean -> noisy: 10 dB holds,
    # 0 dB fails. This is the SNR the model is reliable *down to* -- it pairs with
    # operating_floor (the next, failing SNR = 0 dB) without contradiction.
    assert reliable_floor(pts, threshold=0.8) == 10.0
    assert operating_floor(pts, threshold=0.8) == 0.0
    assert reliable_floor(pts, threshold=0.8) > operating_floor(pts, threshold=0.8)

    # everything holds -> reliable down to the lowest tested numeric SNR
    assert reliable_floor(pts, threshold=0.5) == 0.0
    # even clean fails -> not reliable at any noisy SNR
    assert reliable_floor(pts, threshold=0.99) is None
