"""Front-end float32 vs float64 decision-parity A/B (TDD, red first).

The model runs INT8 on the NPU, but the log-mel *front-end* runs in floating
point on the CPU (``model/features.py``: a MelSpectrogram followed by
``log10(mel + eps)``). ``log10`` of near-zero mel energy in low-signal frames is
the numerically delicate step — single precision could in principle drift far
enough there to move a borderline score across the stress threshold.

No artifact has certified the front-end is well-conditioned. This A/B runs the
SAME labelled noisy waveforms through the front-end in float32 and float64,
feeds both into the same fp32 model, and measures (a) the worst per-window score
divergence and (b) how many *decisions* flip across the threshold. If float32
matches float64 to a hair with zero flips, the front-end is precision-robust and
the on-device single-precision path inherits that — a real certification.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested
without extracting; the heavy build path trains one tiny net and runs the
front-end twice.
"""

from __future__ import annotations

import json

import pytest


def _records(*rows):
    """Build (snr_db, max_abs_diff, n_flips, n) records, clean->noisy."""
    snrs = [None, 20.0, 10.0, 0.0, -5.0]
    return [(snrs[i], d, f, n) for i, (d, f, n) in enumerate(rows)]


# --- pure reduction logic (no extraction) ----------------------------------

def test_reports_one_point_per_snr():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((1e-7, 0, 128), (2e-7, 0, 128)))
    assert len(out.points) == 2
    assert [p.snr_db for p in out.points] == [None, 20.0]


def test_worst_diff_and_total_flips():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((1e-6, 0, 100), (5e-4, 2, 100), (3e-6, 1, 100)))
    assert out.worst_abs_diff == pytest.approx(5e-4)
    assert out.total_flips == 3
    assert out.total_n == 300


def test_flip_rate():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((0.0, 3, 100), (0.0, 0, 50)))
    assert out.flip_rate == pytest.approx(3 / 150)


def test_per_point_flip_rate():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((0.0, 4, 200)))
    assert out.points[0].flip_rate == pytest.approx(0.02)


def test_precision_robust_when_no_flips_and_tiny_diff():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((1e-6, 0, 128), (2e-6, 0, 128)), tol=1e-3)
    assert out.precision_robust is True


def test_not_robust_when_decisions_flip():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((1e-6, 0, 128), (4e-4, 1, 128)), tol=1e-3)
    assert out.precision_robust is False


def test_not_robust_when_diff_exceeds_tol():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((5e-3, 0, 128)), tol=1e-3)
    assert out.precision_robust is False


def test_empty_rejected():
    from model.frontend_precision_ab import frontend_precision_ab

    with pytest.raises(ValueError):
        frontend_precision_ab([])


def test_nonpositive_n_rejected():
    from model.frontend_precision_ab import frontend_precision_ab

    with pytest.raises(ValueError):
        frontend_precision_ab(_records((1e-6, 0, 0)))


def test_verdict_nonempty():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((1e-6, 0, 128)))
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.frontend_precision_ab import frontend_precision_ab

    out = frontend_precision_ab(_records((1e-6, 0, 128), (4e-4, 1, 128)))
    back = json.loads(json.dumps(out.to_dict()))
    assert len(back["points"]) == 2
    assert "worst_abs_diff" in back
    assert "total_flips" in back
    assert "precision_robust" in back


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "frontend_precision_ab.json" in names


# --- heavy build path (trains + runs the front-end at two dtypes) -----------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.frontend_precision_ab import build_frontend_precision_ab

    out = build_frontend_precision_ab(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=8,
        snr_levels=(None, 0.0), out_dir=tmp_path,
    )
    assert len(out.points) == 2
    assert out.total_n > 0
    # float32 vs float64 should agree closely on this well-conditioned front-end.
    assert out.worst_abs_diff < 0.1
    assert (tmp_path / "frontend_precision_ab.json").is_file()
    assert (tmp_path / "frontend_precision_ab.md").is_file()
