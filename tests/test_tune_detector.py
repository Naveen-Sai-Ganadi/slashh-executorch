"""Tests for the detector tuning A/B harness (model/tune_detector.py).

Builds on the host StressDetector (the golden spec) to sweep the on-device UX
knobs — STRESS_THRESHOLD / RELEASE_THRESHOLD / EMA_ALPHA — against labelled
per-window score traces, and recommend an operating point that is both
accurate (latch matches truth) and steady (few flickers). Fully host-side; no
device, no AI Hub.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.tune_detector import (
    ConfigMetrics,
    evaluate_config,
    sweep,
    synthetic_score_trace,
)


def test_synthetic_trace_is_consistent_and_deterministic() -> None:
    scores, voiced, labels = synthetic_score_trace(n_segments=4, seg_len=10, seed=0)
    assert len(scores) == len(voiced) == len(labels) == 40
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert set(labels) <= {0, 1}
    # deterministic for a fixed seed
    again = synthetic_score_trace(n_segments=4, seg_len=10, seed=0)[0]
    assert scores == again
    # different seed -> different trace
    other = synthetic_score_trace(n_segments=4, seg_len=10, seed=1)[0]
    assert scores != other


def test_evaluate_config_reports_accuracy_and_flicker() -> None:
    scores, voiced, labels = synthetic_score_trace(n_segments=4, seg_len=12, seed=0)
    m = evaluate_config(
        scores, voiced, labels,
        stress_threshold=0.6, release_threshold=0.45, ema_alpha=0.4,
    )
    assert isinstance(m, ConfigMetrics)
    assert 0.0 <= m.accuracy <= 1.0
    assert m.flicker >= 0
    assert m.n_windows == len(scores)
    # a well-separated synthetic trace should be classified well above chance
    assert m.accuracy > 0.7


def test_smoothing_reduces_flicker_on_noisy_trace() -> None:
    # noisy scores -> aggressive (alpha=1, no smoothing) flickers more than smoothed
    scores, voiced, labels = synthetic_score_trace(
        n_segments=4, seg_len=15, seed=3, noise=0.3
    )
    raw = evaluate_config(scores, voiced, labels,
                          stress_threshold=0.6, release_threshold=0.45, ema_alpha=1.0)
    smoothed = evaluate_config(scores, voiced, labels,
                               stress_threshold=0.6, release_threshold=0.45, ema_alpha=0.3)
    assert smoothed.flicker <= raw.flicker


def test_sweep_picks_a_best_config_and_writes_artifacts(tmp_path: Path) -> None:
    scores, voiced, labels = synthetic_score_trace(n_segments=6, seg_len=12, seed=0)
    result = sweep(
        scores, voiced, labels,
        stress_thresholds=[0.55, 0.6, 0.65],
        release_thresholds=[0.4, 0.45],
        ema_alphas=[0.3, 0.6, 1.0],
        out_dir=tmp_path,
    )
    # every evaluated config is returned, with the recommended one flagged
    assert len(result.results) > 0
    assert result.best in result.results
    # best maximises the combined score over the grid
    assert all(result.best.score >= r.score for r in result.results)
    # valid band only: release strictly below stress
    assert all(m.release_threshold < m.stress_threshold for m in result.results)

    assert (tmp_path / "detector_tuning.json").is_file()
    assert (tmp_path / "detector_tuning.md").is_file()
    payload = json.loads((tmp_path / "detector_tuning.json").read_text())
    assert "best" in payload and "results" in payload
    assert len(payload["results"]) == len(result.results)
