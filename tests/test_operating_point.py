"""Operating-point / ROC sweep A/B (TDD, red first).

The per-window classifier emits a probability and thresholds it at a fixed 0.5 to
decide stressed/calm (the raw decision the detector's EMA + hysteresis then
smooth). ``tune_detector`` tunes the *hysteresis* knobs; ``base_rate_precision``
measures TPR/FPR at the *single* 0.5 threshold then sweeps prevalence. Neither
sweeps the per-window decision threshold itself to ask the classic question: is
0.5 the right operating point, what's the ROC AUC, and how much accuracy/F1
headroom is left on the table by not moving it?

This A/B sweeps the decision threshold across [0, 1], computes TPR / FPR /
accuracy / F1 / Youden's J at each, integrates the ROC AUC (trapezoidal), and
finds the accuracy-, F1-, and Youden-optimal thresholds — then reports how far
the default 0.5 sits from optimal and whether it is well placed.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set ROC points; the heavy build trains one tiny net and sweeps real scores.
"""

from __future__ import annotations

import json

import pytest

DEFAULT = 0.5


def _pts(*rows):
    """Build (threshold, tpr, fpr, accuracy, f1, n_pos, n_neg) records."""
    return list(rows)


# --- pure reduction logic (no training) ------------------------------------

def test_auc_perfect_classifier_is_one():
    from model.operating_point import operating_point

    # perfect separation at 0.5: a threshold where tpr=1, fpr=0.
    out = operating_point(_pts(
        (0.0, 1.0, 1.0, 0.5, 0.667, 50, 50),
        (0.5, 1.0, 0.0, 1.0, 1.0, 50, 50),
        (1.0, 0.0, 0.0, 0.5, 0.0, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.auc == pytest.approx(1.0)


def test_auc_diagonal_is_half():
    from model.operating_point import operating_point

    out = operating_point(_pts(
        (0.0, 1.0, 1.0, 0.5, 0.667, 50, 50),
        (0.5, 0.5, 0.5, 0.5, 0.5, 50, 50),
        (1.0, 0.0, 0.0, 0.5, 0.0, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.auc == pytest.approx(0.5)


def test_best_accuracy_threshold():
    from model.operating_point import operating_point

    out = operating_point(_pts(
        (0.3, 0.95, 0.30, 0.82, 0.84, 50, 50),
        (0.5, 0.90, 0.10, 0.90, 0.90, 50, 50),
        (0.7, 0.70, 0.02, 0.84, 0.81, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.best_accuracy_threshold == pytest.approx(0.5)
    assert out.best_accuracy == pytest.approx(0.90)


def test_best_youden_threshold():
    from model.operating_point import operating_point

    # Youden's J = tpr - fpr; maximized at threshold 0.5 here (0.90-0.10=0.80).
    out = operating_point(_pts(
        (0.3, 0.95, 0.30, 0.82, 0.84, 50, 50),
        (0.5, 0.90, 0.10, 0.90, 0.90, 50, 50),
        (0.7, 0.70, 0.02, 0.84, 0.81, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.best_youden_threshold == pytest.approx(0.5)
    assert out.best_youden_j == pytest.approx(0.80)


def test_best_f1_threshold():
    from model.operating_point import operating_point

    out = operating_point(_pts(
        (0.3, 0.95, 0.30, 0.82, 0.88, 50, 50),
        (0.5, 0.90, 0.10, 0.90, 0.90, 50, 50),
        (0.7, 0.70, 0.02, 0.84, 0.81, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.best_f1_threshold == pytest.approx(0.5)
    assert out.best_f1 == pytest.approx(0.90)


def test_default_accuracy_and_headroom():
    from model.operating_point import operating_point

    # default 0.5 accuracy 0.84, best is 0.90 at 0.3 -> headroom 0.06.
    out = operating_point(_pts(
        (0.3, 0.95, 0.18, 0.90, 0.90, 50, 50),
        (0.5, 0.80, 0.12, 0.84, 0.83, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.default_accuracy == pytest.approx(0.84)
    assert out.accuracy_headroom == pytest.approx(0.06)
    assert out.well_placed is False


def test_well_placed_true_when_default_near_best():
    from model.operating_point import operating_point

    out = operating_point(_pts(
        (0.3, 0.95, 0.18, 0.895, 0.90, 50, 50),
        (0.5, 0.90, 0.10, 0.90, 0.90, 50, 50),
    ), default_threshold=DEFAULT, headroom_tol=0.02)
    assert out.well_placed is True
    assert out.accuracy_headroom == pytest.approx(0.0)


def test_default_uses_nearest_threshold_when_no_exact():
    from model.operating_point import operating_point

    # no point exactly at 0.5; nearest is 0.45.
    out = operating_point(_pts(
        (0.45, 0.88, 0.12, 0.88, 0.88, 50, 50),
        (0.7, 0.70, 0.02, 0.84, 0.81, 50, 50),
    ), default_threshold=DEFAULT)
    assert out.default_accuracy == pytest.approx(0.88)


def test_empty_rejected():
    from model.operating_point import operating_point

    with pytest.raises(ValueError):
        operating_point([], default_threshold=DEFAULT)


def test_threshold_out_of_range_rejected():
    from model.operating_point import operating_point

    with pytest.raises(ValueError):
        operating_point(_pts((1.5, 0.9, 0.1, 0.9, 0.9, 50, 50)), default_threshold=DEFAULT)


def test_rate_out_of_range_rejected():
    from model.operating_point import operating_point

    with pytest.raises(ValueError):
        operating_point(_pts((0.5, 1.2, 0.1, 0.9, 0.9, 50, 50)), default_threshold=DEFAULT)
    with pytest.raises(ValueError):
        operating_point(_pts((0.5, 0.9, 0.1, 0.9, 0.9, 0, 50)), default_threshold=DEFAULT)


def test_verdict_nonempty():
    from model.operating_point import operating_point

    out = operating_point(_pts(
        (0.5, 0.90, 0.10, 0.90, 0.90, 50, 50),
        (0.3, 0.95, 0.30, 0.82, 0.84, 50, 50),
    ), default_threshold=DEFAULT)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.operating_point import operating_point

    out = operating_point(_pts(
        (0.5, 0.90, 0.10, 0.90, 0.90, 50, 50),
        (0.3, 0.95, 0.30, 0.82, 0.84, 50, 50),
    ), default_threshold=DEFAULT)
    back = json.loads(json.dumps(out.to_dict()))
    assert "auc" in back
    assert "best_accuracy_threshold" in back
    assert "default_accuracy" in back
    assert "accuracy_headroom" in back
    assert "well_placed" in back
    assert len(back["points"]) == 2
    assert "youden_j" in back["points"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "operating_point.json" in names


# --- heavy build path (trains + sweeps real scores) ------------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.operating_point import build_operating_point

    out = build_operating_point(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=16,
        thresholds=(0.3, 0.5, 0.7), out_dir=tmp_path,
    )
    assert len(out.points) == 3
    assert 0.0 <= out.auc <= 1.0
    assert out.default_accuracy is not None
    assert (tmp_path / "operating_point.json").is_file()
    assert (tmp_path / "operating_point.md").is_file()
