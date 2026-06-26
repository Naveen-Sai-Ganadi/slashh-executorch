"""Base-rate (prior-shift) precision A/B (TDD, red first).

Every accuracy/F1/robustness artifact here evaluates on a *balanced* set — 50%
stressed, 50% calm. Real usage is not balanced: a phone listening through a day
hears overwhelmingly calm audio. Under that skew even a small per-window false-
alarm rate dominates the alarms the user actually sees, because calm windows
vastly outnumber stressed ones (the base-rate fallacy).

This A/B takes the model's per-window TPR/FPR at the operating threshold and,
via Bayes, reports the realized **precision** of an alarm across a sweep of
stress prevalences, plus the break-even prevalence where precision crosses 0.5
(below it most alarms are false). It is the "is the detector usable in the wild"
number that balanced accuracy hides.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set TPR/FPR; the heavy build path trains one tiny net and measures TPR/FPR.
"""

from __future__ import annotations

import json

import pytest


# --- pure reduction logic (no training) ------------------------------------

def test_precision_at_balanced_prior_matches_bayes():
    from model.base_rate_precision import base_rate_precision

    # tpr=0.9, fpr=0.1 at 50/50: precision = .9*.5/(.9*.5+.1*.5) = 0.9
    out = base_rate_precision([(None, 0.9, 0.1, 100)], priors=(0.5,))
    assert out.points[0].precision_at(0.5) == pytest.approx(0.9)


def test_precision_collapses_at_low_prior():
    from model.base_rate_precision import base_rate_precision

    # tpr=0.9, fpr=0.1 at 1% prevalence: .9*.01/(.9*.01+.1*.99) ~ 0.0833
    out = base_rate_precision([(None, 0.9, 0.1, 100)], priors=(0.01,))
    assert out.points[0].precision_at(0.01) == pytest.approx(0.9 * 0.01 / (0.9 * 0.01 + 0.1 * 0.99))


def test_break_even_prior():
    from model.base_rate_precision import base_rate_precision

    # precision = 0.5 when tpr*p = fpr*(1-p) -> p = fpr/(tpr+fpr)
    out = base_rate_precision([(None, 0.8, 0.2, 100)], priors=(0.5,))
    assert out.points[0].break_even_prior == pytest.approx(0.2 / (0.8 + 0.2))


def test_perfect_specificity_has_zero_break_even():
    from model.base_rate_precision import base_rate_precision

    # fpr=0 -> precision is 1.0 at any positive prevalence -> break-even at 0.
    out = base_rate_precision([(None, 0.9, 0.0, 100)], priors=(0.001,))
    assert out.points[0].break_even_prior == pytest.approx(0.0)
    assert out.points[0].precision_at(0.001) == pytest.approx(1.0)


def test_no_positive_predictions_is_none():
    from model.base_rate_precision import base_rate_precision

    # tpr=fpr=0: nothing is ever flagged -> precision undefined, break-even None.
    out = base_rate_precision([(None, 0.0, 0.0, 100)], priors=(0.1,))
    assert out.points[0].precision_at(0.1) is None
    assert out.points[0].break_even_prior is None


def test_worst_break_even_and_realistic_precision():
    from model.base_rate_precision import base_rate_precision

    pts = [(None, 0.9, 0.05, 100), (-5.0, 0.9, 0.4, 100)]
    out = base_rate_precision(pts, priors=(0.5, 0.1, 0.02))
    # noisier SNR needs a higher prevalence to break even -> that's the worst.
    assert out.worst_break_even == pytest.approx(0.4 / (0.9 + 0.4))
    # realistic prior is the smallest swept prior.
    assert out.realistic_prior == 0.02


def test_usable_flag_is_bool():
    from model.base_rate_precision import base_rate_precision

    out = base_rate_precision([(None, 0.9, 0.05, 100)], priors=(0.5, 0.1))
    assert isinstance(out.usable_at_realistic, bool)


def test_empty_rejected():
    from model.base_rate_precision import base_rate_precision

    with pytest.raises(ValueError):
        base_rate_precision([], priors=(0.1,))


def test_empty_priors_rejected():
    from model.base_rate_precision import base_rate_precision

    with pytest.raises(ValueError):
        base_rate_precision([(None, 0.9, 0.1, 100)], priors=())


def test_priors_out_of_range_rejected():
    from model.base_rate_precision import base_rate_precision

    with pytest.raises(ValueError):
        base_rate_precision([(None, 0.9, 0.1, 100)], priors=(0.0, 0.5))
    with pytest.raises(ValueError):
        base_rate_precision([(None, 0.9, 0.1, 100)], priors=(0.5, 1.0))


def test_verdict_nonempty():
    from model.base_rate_precision import base_rate_precision

    out = base_rate_precision([(None, 0.9, 0.1, 100)], priors=(0.5, 0.1))
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.base_rate_precision import base_rate_precision

    out = base_rate_precision([(None, 0.9, 0.1, 100), (-5.0, 0.9, 0.3, 100)], priors=(0.5, 0.1))
    back = json.loads(json.dumps(out.to_dict()))
    assert len(back["points"]) == 2
    assert "worst_break_even" in back
    assert "realistic_prior" in back
    assert "usable_at_realistic" in back
    # precision grid serialized per point
    assert "precision_by_prior" in back["points"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "base_rate_precision.json" in names


# --- heavy build path (trains + measures TPR/FPR) ---------------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.base_rate_precision import build_base_rate_precision

    out = build_base_rate_precision(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=16,
        snr_levels=(None, 0.0), priors=(0.5, 0.1, 0.02), out_dir=tmp_path,
    )
    assert len(out.points) == 2
    for p in out.points:
        assert 0.0 <= p.tpr <= 1.0
        assert 0.0 <= p.fpr <= 1.0
    assert (tmp_path / "base_rate_precision.json").is_file()
    assert (tmp_path / "base_rate_precision.md").is_file()
