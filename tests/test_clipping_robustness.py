"""Clipping / saturation robustness A/B (TDD, red first).

``gain_robustness`` swept input *level* with pure scaling — no clipping — and
found the loud end tolerated to +36 dB. But real microphones / ADCs **clip** at
full scale: loud or close-talking capture flat-tops the waveform, a nonlinear
distortion (harmonics, lost peaks) that neither additive-noise SNR nor linear
gain models. This A/B isolates that distortion: it hard-clips the waveform to a
fraction of its own peak, then **rescales back to the original peak** so the
*level* is unchanged and only the flat-topping remains. It sweeps clip severity,
re-extracts log-mel features, and measures accuracy — the clipping the model
tolerates before saturation breaks detection.

clip_ratio = fraction of peak amplitude retained before clamping: 1.0 = no clip,
0.1 = clamp to 10% of peak (heavy saturation). clipped_fraction is the share of
samples that actually hit the clip, reported for context.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set accuracies; the heavy build trains one tiny net and clips eval sets.
"""

from __future__ import annotations

import json

import pytest

BAR = 0.8


def _pts(*rows):
    """Build (clip_ratio, accuracy, clipped_fraction, n) records."""
    return list(rows)


# --- pure reduction logic (no training) ------------------------------------

def test_clean_accuracy_picks_unclipped():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.97, 0.0, 50), (0.5, 0.9, 0.2, 50)), accuracy_bar=BAR)
    assert out.clean_accuracy == pytest.approx(0.97)


def test_clip_tolerant_true_when_all_clear_bar():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.95, 0.0, 50), (0.5, 0.9, 0.2, 50), (0.1, 0.85, 0.6, 50)),
        accuracy_bar=BAR)
    assert out.clip_tolerant is True


def test_clip_tolerant_false_when_one_fails():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.95, 0.0, 50), (0.1, 0.6, 0.6, 50)), accuracy_bar=BAR)
    assert out.clip_tolerant is False


def test_reliable_floor_is_lowest_contiguous_ratio():
    from model.clipping_robustness import clipping_robustness

    # reliable from 1.0 down through 0.25, breaks at 0.1.
    out = clipping_robustness(
        _pts((1.0, 0.95, 0.0, 50), (0.5, 0.9, 0.2, 50),
             (0.25, 0.85, 0.4, 50), (0.1, 0.6, 0.7, 50)),
        accuracy_bar=BAR)
    assert out.reliable_floor_ratio == pytest.approx(0.25)


def test_reliable_floor_none_when_clean_fails():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.5, 0.0, 50), (0.5, 0.9, 0.2, 50)), accuracy_bar=BAR)
    assert out.reliable_floor_ratio is None


def test_worst_accuracy_and_ratio():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.95, 0.0, 50), (0.1, 0.6, 0.7, 50)), accuracy_bar=BAR)
    assert out.worst_accuracy == pytest.approx(0.6)
    assert out.worst_clip_ratio == pytest.approx(0.1)


def test_no_clean_point_has_none_clean_accuracy():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(_pts((0.5, 0.9, 0.2, 50)), accuracy_bar=BAR)
    assert out.clean_accuracy is None
    assert out.reliable_floor_ratio is None


def test_empty_rejected():
    from model.clipping_robustness import clipping_robustness

    with pytest.raises(ValueError):
        clipping_robustness([], accuracy_bar=BAR)


def test_ratio_out_of_range_rejected():
    from model.clipping_robustness import clipping_robustness

    with pytest.raises(ValueError):
        clipping_robustness(_pts((1.5, 0.9, 0.0, 50)), accuracy_bar=BAR)
    with pytest.raises(ValueError):
        clipping_robustness(_pts((0.0, 0.9, 0.0, 50)), accuracy_bar=BAR)


def test_accuracy_out_of_range_rejected():
    from model.clipping_robustness import clipping_robustness

    with pytest.raises(ValueError):
        clipping_robustness(_pts((1.0, 1.5, 0.0, 50)), accuracy_bar=BAR)


def test_clipped_fraction_out_of_range_rejected():
    from model.clipping_robustness import clipping_robustness

    with pytest.raises(ValueError):
        clipping_robustness(_pts((1.0, 0.9, 1.2, 50)), accuracy_bar=BAR)


def test_nonpositive_n_rejected():
    from model.clipping_robustness import clipping_robustness

    with pytest.raises(ValueError):
        clipping_robustness(_pts((1.0, 0.9, 0.0, 0)), accuracy_bar=BAR)


def test_verdict_nonempty():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.95, 0.0, 50), (0.25, 0.88, 0.4, 50)), accuracy_bar=BAR)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.clipping_robustness import clipping_robustness

    out = clipping_robustness(
        _pts((1.0, 0.95, 0.0, 50), (0.25, 0.88, 0.4, 50)), accuracy_bar=BAR)
    back = json.loads(json.dumps(out.to_dict()))
    assert "clip_tolerant" in back
    assert "reliable_floor_ratio" in back
    assert "clean_accuracy" in back
    assert len(back["points"]) == 2
    assert "clipped_fraction" in back["points"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "clipping_robustness.json" in names


# --- heavy build path (trains + clips eval sets) ---------------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.clipping_robustness import build_clipping_robustness

    out = build_clipping_robustness(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=16,
        clip_ratios=(1.0, 0.25), eval_snr=None,
        out_dir=tmp_path,
    )
    assert len(out.points) == 2
    assert out.clean_accuracy is not None
    assert (tmp_path / "clipping_robustness.json").is_file()
    assert (tmp_path / "clipping_robustness.md").is_file()
