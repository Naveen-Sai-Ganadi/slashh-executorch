"""Tests for the noise failure-mode breakdown (model/noise_failure_mode.py).

The aggregate accuracy-vs-SNR curve (model/robustness.py) tells you *that* the
model degrades under noise, but not *how*. For a stress detector the direction
matters: missing a stressed window (false negative — the detector goes silent)
is a very different product failure from a false alarm (false positive — it
cries wolf), and they call for opposite mitigations at the detector gate. This
harness splits each SNR into precision/recall + which way the errors lean, and
names the dominant failure at the first SNR that drops below the threshold.
Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.model import StressNet
from model.noise_failure_mode import (
    DIRECTIONS,
    FailureModeResult,
    noise_failure_mode,
)
from model.train import train


def _trained(seed: int = 0) -> StressNet:
    net = StressNet()
    train(data_dir=None, epochs=10, batch_size=16, lr=1e-3, n_per_class=48,
          seed=seed, model=net)
    return net.eval()


def test_points_carry_valid_directions_and_consistent_counts() -> None:
    model = _trained(seed=0)
    out = noise_failure_mode(
        model, snr_levels=[None, 10.0, 0.0, -10.0], n_per_class=32, seed=1,
    )
    assert isinstance(out, FailureModeResult)
    assert [p.snr_db for p in out.points] == [None, 10.0, 0.0, -10.0]
    for p in out.points:
        assert p.direction in DIRECTIONS
        # precision/recall in [0, 1]; counts non-negative and sum to n
        assert 0.0 <= p.precision <= 1.0 and 0.0 <= p.recall <= 1.0
        assert p.fp >= 0 and p.fn >= 0
        assert p.tp + p.tn + p.fp + p.fn == p.n
        # no errors of either kind => balanced
        if p.fp == 0 and p.fn == 0:
            assert p.direction == "balanced"


def test_clean_point_matches_aggregate_accuracy() -> None:
    # the clean (no-noise) point reproduces the plain robustness accuracy, since
    # both run the same dataset draw; this anchors the breakdown to the curve.
    from model.robustness import robustness_curve

    model = _trained(seed=0)
    fm = noise_failure_mode(model, snr_levels=[None, 0.0], n_per_class=32, seed=1)
    rc = robustness_curve(model, snr_levels=[None, 0.0], n_per_class=32, seed=1)
    fm_clean = {p.snr_db: p.accuracy for p in fm.points}[None]
    rc_clean = {p.snr_db: p.accuracy for p in rc.points}[None]
    assert abs(fm_clean - rc_clean) < 1e-9


def test_dominant_failure_matches_first_subthreshold_point() -> None:
    model = _trained(seed=0)
    out = noise_failure_mode(
        model, snr_levels=[None, 10.0, 0.0, -10.0, -20.0], n_per_class=32,
        seed=1, threshold=0.8,
    )
    # find the first point (clean->noisy) below threshold by hand
    first_bad = next((p for p in out.points if p.accuracy < 0.8), None)
    if first_bad is None:
        assert out.dominant_failure is None
    else:
        assert out.dominant_failure == first_bad.direction
        assert out.dominant_failure in DIRECTIONS


def test_multi_seed_sums_counts_and_is_deterministic() -> None:
    model = _trained(seed=0)
    a = noise_failure_mode(
        model, snr_levels=[None, 0.0], n_per_class=24, seed=1,
        eval_seeds=[1, 2, 3],
    )
    b = noise_failure_mode(
        model, snr_levels=[None, 0.0], n_per_class=24, seed=1,
        eval_seeds=[1, 2, 3],
    )
    # deterministic across identical calls
    assert [p.to_dict() for p in a.points] == [p.to_dict() for p in b.points]
    # counts reflect all three seeds (n is summed), each seed is 2*n_per_class
    for p in a.points:
        assert p.n == 3 * (2 * 24)


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    model = _trained(seed=0)
    noise_failure_mode(
        model, snr_levels=[None, 10.0, 0.0], n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    assert (tmp_path / "noise_failure_mode.json").is_file()
    assert (tmp_path / "noise_failure_mode.md").is_file()
    payload = json.loads((tmp_path / "noise_failure_mode.json").read_text())
    assert "points" in payload and "dominant_failure" in payload
    md = (tmp_path / "noise_failure_mode.md").read_text()
    # the table reports precision and recall, not just accuracy
    assert "precision" in md.lower() and "recall" in md.lower()
