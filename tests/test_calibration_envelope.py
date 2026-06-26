"""Cross-initialization calibration-temperature envelope (TDD, red first).

``model/confidence.py`` ships a single hard constant ``DEFAULT_TEMPERATURE =
0.41``. That number was fit by ``model.calibration`` on *one* initialization
(seed 0) of the production net. But this project's whole robustness story exists
because a 1,549-param net is init-sensitive (see ``init_envelope`` /
``recipe_envelope``): a quantity fit on a single seed may be seed-luck. This
harness re-fits the temperature across several independent inits and asks whether
0.41 is a stable default or a draw artifact.

Host-only; no device, no AI Hub token. The heavy ``build_*`` path trains a few
tiny nets and is kept minimal; the reduction logic is unit-tested without
training.
"""

from __future__ import annotations

import json

import pytest


# --- pure reduction logic (no training) -----------------------------------

def _fit(seed, temperature, ece=0.24, ece_after=0.10):
    from model.calibration_envelope import CalibrationFit

    return CalibrationFit(
        seed=seed, temperature=temperature, ece=ece, ece_after_temp=ece_after
    )


def test_envelope_reports_temperature_spread():
    from model.calibration_envelope import calibration_envelope

    fits = [_fit(0, 0.40), _fit(1, 0.45), _fit(2, 0.38)]
    out = calibration_envelope(fits, default_temperature=0.41)

    d = out.to_dict()
    assert d["n_inits"] == 3
    assert d["temp_min"] == pytest.approx(0.38)
    assert d["temp_max"] == pytest.approx(0.45)
    assert d["temp_median"] == pytest.approx(0.40)
    assert d["temp_mean"] == pytest.approx((0.40 + 0.45 + 0.38) / 3)


def test_default_within_spread_true_when_bracketed():
    from model.calibration_envelope import calibration_envelope

    out = calibration_envelope([_fit(0, 0.38), _fit(1, 0.45)], default_temperature=0.41)
    assert out.default_within_spread is True


def test_default_within_spread_false_when_outside():
    from model.calibration_envelope import calibration_envelope

    # every init wants a much sharper temperature than the shipped 0.41
    out = calibration_envelope([_fit(0, 0.20), _fit(1, 0.25)], default_temperature=0.41)
    assert out.default_within_spread is False


def test_all_same_direction_true_when_all_under_confident():
    from model.calibration_envelope import calibration_envelope

    out = calibration_envelope([_fit(0, 0.40), _fit(1, 0.45), _fit(2, 0.50)], default_temperature=0.41)
    # T<1 across the board => every init says "under-confident" (same direction)
    assert out.all_same_direction is True


def test_all_same_direction_false_when_mixed():
    from model.calibration_envelope import calibration_envelope

    out = calibration_envelope([_fit(0, 0.40), _fit(1, 1.30)], default_temperature=0.41)
    assert out.all_same_direction is False


def test_verdict_stable_when_tight_and_bracketed():
    from model.calibration_envelope import calibration_envelope

    out = calibration_envelope([_fit(0, 0.40), _fit(1, 0.42), _fit(2, 0.41)], default_temperature=0.41)
    # tight spread, same direction, default bracketed -> a stable default
    assert "stable" in out.verdict.lower()


def test_empty_fits_rejected():
    from model.calibration_envelope import calibration_envelope

    with pytest.raises(ValueError):
        calibration_envelope([], default_temperature=0.41)


def test_to_dict_round_trips_json():
    from model.calibration_envelope import calibration_envelope

    out = calibration_envelope([_fit(0, 0.40), _fit(1, 0.45)], default_temperature=0.41)
    s = json.dumps(out.to_dict())
    back = json.loads(s)
    assert back["n_inits"] == 2
    assert len(back["fits"]) == 2
    assert back["default_temperature"] == pytest.approx(0.41)


def test_default_temperature_defaults_to_shipped_constant():
    from model.calibration_envelope import calibration_envelope
    from model.confidence import DEFAULT_TEMPERATURE

    out = calibration_envelope([_fit(0, 0.40), _fit(1, 0.45)])
    assert out.default_temperature == pytest.approx(DEFAULT_TEMPERATURE)


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "calibration_envelope.json" in names


# --- heavy build path (tiny: trains a couple of small nets) -----------------

def test_build_trains_inits_and_writes_artifacts(tmp_path):
    from model.calibration_envelope import build_calibration_envelope

    out = build_calibration_envelope(
        seeds=(0, 1), epochs=2, n_per_class=8, eval_n_per_class=8,
        snr_levels=(None, 0.0), out_dir=tmp_path,
    )
    assert len(out.fits) == 2
    assert {f.seed for f in out.fits} == {0, 1}
    for f in out.fits:
        assert f.temperature > 0.0
    assert (tmp_path / "calibration_envelope.json").is_file()
    assert (tmp_path / "calibration_envelope.md").is_file()
