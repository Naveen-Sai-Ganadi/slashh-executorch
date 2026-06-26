"""Tests for the score-calibration analysis (model/calibration.py).

StressNet outputs ``sigmoid(logit)`` in [0,1] and the detector gates on
``STRESS_THRESHOLD = 0.6``. That only makes sense if the score is a trustworthy
*probability* — i.e. of the windows scored 0.6, ~60% are truly stressed. This
harness measures Expected/Max Calibration Error, the Brier score, a reliability
table, and whether one-parameter temperature scaling would help, then renders a
verdict (well-calibrated / over- / under-confident).

The reliability math is exercised on CONSTRUCTED score/label sets with known
answers (perfectly calibrated, perfectly confident, pathologically
overconfident) so the assertions are exact and machine-independent; a small
real-model integration test confirms the collection path. Host-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.audio_config import STRESS_THRESHOLD
from model.calibration import (
    CalibrationResult,
    collect_scores,
    compute_calibration,
)
from model.production import build_production_model


def test_perfectly_calibrated_is_near_zero_ece() -> None:
    # 200 samples all scored 0.5; exactly half are positive. Confidence 0.5,
    # accuracy 0.5 -> ECE ~ 0.
    scores = torch.full((200,), 0.5)
    labels = torch.zeros(200)
    labels[:100] = 1.0
    out = compute_calibration(scores, labels, n_bins=10, fit_temperature=False)
    assert isinstance(out, CalibrationResult)
    assert out.ece <= 1e-6
    assert out.verdict == "well-calibrated"


def test_perfectly_confident_and_correct() -> None:
    # Confident and right: positives at 0.99, negatives at 0.01.
    scores = torch.cat([torch.full((100,), 0.99), torch.full((100,), 0.01)])
    labels = torch.cat([torch.ones(100), torch.zeros(100)])
    out = compute_calibration(scores, labels, n_bins=10, fit_temperature=False)
    assert out.accuracy == 1.0
    assert out.ece <= 0.02  # confidence ~0.99 vs accuracy 1.0


def test_pathological_overconfidence_flagged() -> None:
    # Always says 0.99 but is right only half the time -> big gap, overconfident.
    gen = torch.Generator().manual_seed(0)
    scores = torch.full((400,), 0.99)
    labels = (torch.rand(400, generator=gen) < 0.5).float()
    out = compute_calibration(scores, labels, n_bins=10, fit_temperature=True)
    assert out.ece >= 0.3
    assert out.verdict == "overconfident"
    # temperature scaling should soften the over-confidence, not worsen ECE
    assert out.temperature > 1.0
    assert out.ece_after_temp <= out.ece + 1e-6


def test_metrics_are_bounded_and_bins_partition() -> None:
    gen = torch.Generator().manual_seed(1)
    scores = torch.rand(300, generator=gen)
    labels = (torch.rand(300, generator=gen) < scores).float()
    out = compute_calibration(scores, labels, n_bins=10)
    assert 0.0 <= out.ece <= 1.0
    assert 0.0 <= out.mce <= 1.0
    assert 0.0 <= out.brier <= 1.0
    assert out.mce >= out.ece - 1e-9  # max error >= mean error
    assert sum(b.count for b in out.bins) == 300
    assert out.threshold == STRESS_THRESHOLD


def test_verdict_is_one_of_three() -> None:
    gen = torch.Generator().manual_seed(2)
    scores = torch.rand(200, generator=gen)
    labels = (torch.rand(200, generator=gen) < scores).float()
    out = compute_calibration(scores, labels)
    assert out.verdict in {"well-calibrated", "overconfident", "underconfident"}


def test_collect_scores_from_real_model() -> None:
    model = build_production_model().eval()
    scores, labels = collect_scores(
        model, snr_levels=[None, 0.0], n_per_class=16, seed=1,
    )
    assert scores.shape == labels.shape
    assert scores.numel() == labels.numel() > 0
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
    assert set(int(v) for v in labels.unique().tolist()) <= {0, 1}


def test_to_dict_and_write(tmp_path: Path) -> None:
    scores = torch.cat([torch.full((50,), 0.8), torch.full((50,), 0.2)])
    labels = torch.cat([torch.ones(50), torch.zeros(50)])
    out = compute_calibration(scores, labels, out_dir=tmp_path)
    d = out.to_dict()
    for k in ("ece", "mce", "brier", "accuracy", "bins", "temperature",
              "ece_after_temp", "threshold", "verdict"):
        assert k in d
    assert json.loads(json.dumps(d)) == d
    assert (tmp_path / "calibration.json").is_file()
    assert (tmp_path / "calibration.md").is_file()
    md = (tmp_path / "calibration.md").read_text().lower()
    assert "calibration" in md and "ece" in md


def test_deterministic() -> None:
    gen = torch.Generator().manual_seed(3)
    scores = torch.rand(150, generator=gen)
    gen2 = torch.Generator().manual_seed(3)
    labels = (torch.rand(150, generator=gen2) < 0.5).float()
    a = compute_calibration(scores, labels).to_dict()
    b = compute_calibration(scores, labels).to_dict()
    assert a == b
