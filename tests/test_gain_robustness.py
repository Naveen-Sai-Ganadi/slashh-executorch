"""Input-gain (level) robustness A/B (TDD, red first).

Every robustness artifact so far varies the *signal-to-noise ratio* (additive
noise at a fixed signal level). None varies the absolute input *level*. That
matters here because the log-mel front-end (model/features.py) is
``log10(mel + eps)`` with **no per-window level normalization** — no mean
subtraction, no peak-norm. Scaling the captured waveform by a gain ``g`` scales
the mel power by ``g**2``, i.e. shifts every (non-floored) log-mel bin by a
constant ``2*log10(g)``. So how loudly the user speaks, or the device mic gain,
feeds the model a DC-shifted feature map it was never normalized against. A
1,549-param conv net may or may not have learned to ignore that shift.

This A/B sweeps an input-gain range (dB), re-extracts features from gained
waveforms (so the eps-floor nonlinearity is faithful, not a feature-domain
approximation), and measures accuracy at each level. It reports the contiguous
gain band around unity over which detection stays reliable — the dynamic range
the model tolerates before quiet or loud speech breaks it.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set accuracies; the heavy build trains one tiny net and evaluates gained sets.
"""

from __future__ import annotations

import json

import pytest

BAR = 0.8


def _pts(*rows):
    """Build (gain_db, accuracy, n) records."""
    return list(rows)


# --- pure reduction logic (no training) ------------------------------------

def test_gain_factor_conversion():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((0.0, 0.95, 50), (20.0, 0.9, 50)), accuracy_bar=BAR)
    by_db = {p.gain_db: p for p in out.points}
    assert by_db[0.0].gain_factor == pytest.approx(1.0)
    assert by_db[20.0].gain_factor == pytest.approx(10.0)


def test_unity_accuracy_picks_zero_db():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-6.0, 0.9, 50), (0.0, 0.97, 50), (6.0, 0.88, 50)),
                          accuracy_bar=BAR)
    assert out.unity_accuracy == pytest.approx(0.97)


def test_level_invariant_true_when_all_clear_bar():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-12.0, 0.85, 50), (0.0, 0.95, 50), (12.0, 0.82, 50)),
                          accuracy_bar=BAR)
    assert out.level_invariant is True


def test_level_invariant_false_when_one_fails():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-12.0, 0.6, 50), (0.0, 0.95, 50), (12.0, 0.82, 50)),
                          accuracy_bar=BAR)
    assert out.level_invariant is False


def test_reliable_band_is_contiguous_around_unity():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(
        _pts((-12.0, 0.7, 50), (-6.0, 0.9, 50), (0.0, 0.95, 50),
             (6.0, 0.9, 50), (12.0, 0.6, 50)),
        accuracy_bar=BAR,
    )
    assert out.reliable_low_db == pytest.approx(-6.0)
    assert out.reliable_high_db == pytest.approx(6.0)
    assert out.usable_band_db == pytest.approx(12.0)


def test_worst_accuracy_and_gain():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(
        _pts((-12.0, 0.7, 50), (0.0, 0.95, 50), (12.0, 0.6, 50)),
        accuracy_bar=BAR,
    )
    assert out.worst_accuracy == pytest.approx(0.6)
    assert out.worst_gain_db == pytest.approx(12.0)


def test_band_none_when_unity_fails():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-6.0, 0.9, 50), (0.0, 0.5, 50), (6.0, 0.9, 50)),
                          accuracy_bar=BAR)
    assert out.reliable_low_db is None
    assert out.reliable_high_db is None
    assert out.level_invariant is False


def test_no_unity_point_has_none_unity_accuracy():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-6.0, 0.9, 50), (6.0, 0.9, 50)), accuracy_bar=BAR)
    assert out.unity_accuracy is None
    assert out.reliable_low_db is None


def test_empty_rejected():
    from model.gain_robustness import gain_robustness

    with pytest.raises(ValueError):
        gain_robustness([], accuracy_bar=BAR)


def test_accuracy_out_of_range_rejected():
    from model.gain_robustness import gain_robustness

    with pytest.raises(ValueError):
        gain_robustness(_pts((0.0, 1.5, 50)), accuracy_bar=BAR)


def test_nonpositive_n_rejected():
    from model.gain_robustness import gain_robustness

    with pytest.raises(ValueError):
        gain_robustness(_pts((0.0, 0.9, 0)), accuracy_bar=BAR)


def test_verdict_nonempty():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-6.0, 0.9, 50), (0.0, 0.95, 50), (6.0, 0.9, 50)),
                          accuracy_bar=BAR)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.gain_robustness import gain_robustness

    out = gain_robustness(_pts((-6.0, 0.9, 50), (0.0, 0.95, 50), (6.0, 0.88, 50)),
                          accuracy_bar=BAR)
    back = json.loads(json.dumps(out.to_dict()))
    assert "level_invariant" in back
    assert "reliable_low_db" in back
    assert "unity_accuracy" in back
    assert len(back["points"]) == 3
    assert "gain_factor" in back["points"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "gain_robustness.json" in names


# --- heavy build path (trains + evaluates gained sets) ---------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.gain_robustness import build_gain_robustness

    out = build_gain_robustness(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=16,
        gain_db_levels=(-12.0, 0.0, 12.0), eval_snr=None,
        out_dir=tmp_path,
    )
    assert len(out.points) == 3
    assert out.unity_accuracy is not None
    assert (tmp_path / "gain_robustness.json").is_file()
    assert (tmp_path / "gain_robustness.md").is_file()
