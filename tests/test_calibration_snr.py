"""Per-SNR calibration breakdown (TDD, red first).

``model.calibration`` pools scores across the whole noise sweep into a single
ECE. That can hide *where* the calibration lives: the confidence read-out can be
trustworthy on clean audio yet meaningless at the noisy reliable floor (-5 dB),
exactly where the user is most likely to need it. This harness re-computes
calibration *per SNR* and asks whether the confidence number can be trusted
across the operating range — especially at the floor — or only in the quiet.

Host-only; no device, no AI Hub token. Pure reduction is unit-tested without
training; the heavy ``build_*`` path trains one tiny net.
"""

from __future__ import annotations

import json

import pytest
import torch


# --- pure reduction logic (no training) -----------------------------------

def _triple(snr, n, hedge, *, seed, flip):
    """One (snr_db, scores, labels) tuple with tunable hedge + mislabels."""
    g = torch.Generator().manual_seed(seed)
    labels = (torch.arange(n) % 2).float()
    base = labels.clone()
    if flip:
        base[:flip] = 1.0 - base[:flip]
    scores = 0.5 + (base - 0.5) * hedge
    scores = (scores + 0.02 * torch.randn(n, generator=g)).clamp(0.01, 0.99)
    return (snr, scores, labels)


def _sweep():
    # clean -> noisier: accuracy falls (more flips) and scores hedge to 0.5.
    return [
        _triple(None, 120, 0.8, seed=0, flip=4),
        _triple(20.0, 120, 0.7, seed=1, flip=12),
        _triple(0.0, 120, 0.55, seed=2, flip=30),
        _triple(-5.0, 120, 0.5, seed=3, flip=52),
    ]


def test_reports_one_row_per_snr():
    from model.calibration_snr import calibration_snr

    out = calibration_snr(_sweep())
    assert len(out.per_snr) == 4
    snrs = [r.snr_db for r in out.per_snr]
    assert snrs == [None, 20.0, 0.0, -5.0]
    for r in out.per_snr:
        assert r.n == 120
        assert 0.0 <= r.accuracy <= 1.0
        assert r.ece >= 0.0


def test_per_snr_ece_matches_compute_calibration():
    from model.calibration_snr import calibration_snr
    from model.calibration import compute_calibration

    sweep = _sweep()
    out = calibration_snr(sweep)
    for (snr, s, y), row in zip(sweep, out.per_snr):
        assert row.ece == pytest.approx(
            compute_calibration(s, y, fit_temperature=False).ece, abs=1e-9
        )


def test_global_temperature_defaults_to_constant():
    from model.calibration_snr import calibration_snr
    from model.confidence import DEFAULT_TEMPERATURE

    out = calibration_snr(_sweep())
    assert out.global_temperature == pytest.approx(DEFAULT_TEMPERATURE)


def test_floor_row_is_the_lowest_snr():
    from model.calibration_snr import calibration_snr

    out = calibration_snr(_sweep())
    # clean (None) sorts as "least noisy"; -5 dB is the floor.
    assert out.floor.snr_db == -5.0


def test_floor_trustworthy_is_bool():
    from model.calibration_snr import calibration_snr

    out = calibration_snr(_sweep())
    assert isinstance(out.floor_trustworthy, bool)


def test_clean_floor_is_trustworthy():
    from model.calibration_snr import calibration_snr

    # a sweep whose "floor" is still high-accuracy -> trustworthy at the floor
    sweep = [
        _triple(None, 120, 0.8, seed=0, flip=4),
        _triple(0.0, 120, 0.78, seed=1, flip=6),
    ]
    out = calibration_snr(sweep)
    assert out.floor_trustworthy is True


def test_noise_floor_untrustworthy():
    from model.calibration_snr import calibration_snr

    # floor is near-chance accuracy -> the confidence number is meaningless there
    sweep = [
        _triple(None, 120, 0.8, seed=0, flip=4),
        _triple(-5.0, 120, 0.5, seed=3, flip=58),
    ]
    out = calibration_snr(sweep)
    assert out.floor_trustworthy is False


def test_verdict_is_nonempty_str():
    from model.calibration_snr import calibration_snr

    out = calibration_snr(_sweep())
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_empty_rejected():
    from model.calibration_snr import calibration_snr

    with pytest.raises(ValueError):
        calibration_snr([])


def test_to_dict_round_trips_json():
    from model.calibration_snr import calibration_snr

    out = calibration_snr(_sweep())
    s = json.dumps(out.to_dict())
    back = json.loads(s)
    assert len(back["per_snr"]) == 4
    assert "floor_trustworthy" in back
    assert back["per_snr"][0]["snr_db"] is None


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "calibration_snr.json" in names


# --- heavy build path (tiny: trains one small net) --------------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.calibration_snr import build_calibration_snr

    out = build_calibration_snr(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=8,
        snr_levels=(None, 0.0), out_dir=tmp_path,
    )
    assert len(out.per_snr) == 2
    assert (tmp_path / "calibration_snr.json").is_file()
    assert (tmp_path / "calibration_snr.md").is_file()
