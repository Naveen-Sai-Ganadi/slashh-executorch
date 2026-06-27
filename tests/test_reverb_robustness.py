"""Reverberation robustness A/B (TDD, red first).

The robustness suite already covers *additive* noise (``robustness``), *nonlinear*
clipping (``clipping_robustness``), and *linear* level (``gain_robustness``). The
one common real-world distortion still unmodeled is **convolutive**: room reverb.
Hands-free / speakerphone / across-the-room capture convolves the voice with a
room impulse response, smearing the temporal envelope (late reflections, a decay
tail) — a distortion none of the additive/nonlinear/level axes reproduce. This
A/B isolates it: each waveform is convolved with a synthetic exponential-decay
RIR parameterized by RT60, then **rescaled back to the original RMS** so the gross
level/energy is unchanged and only the temporal smearing remains. It sweeps RT60,
re-extracts log-mel features, and measures accuracy — the reverberation the model
tolerates before the smeared envelope breaks detection.

rt60_s = reverberation time in seconds (0.0 = dry / anechoic reference; 0.6 ≈ a
live room; 1.0 ≈ a hall). tail_fraction is the share of output energy in the
reverberant tail (beyond the direct path), reported for context.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set accuracies; the heavy build trains one tiny net and reverberates eval
sets.
"""

from __future__ import annotations

import json

import pytest

BAR = 0.8


def _pts(*rows):
    """Build (rt60_s, accuracy, tail_fraction, n) records."""
    return list(rows)


# --- pure reduction logic (no training) ------------------------------------

def test_clean_accuracy_picks_dry():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.97, 0.0, 50), (0.3, 0.9, 0.4, 50)), accuracy_bar=BAR)
    assert out.clean_accuracy == pytest.approx(0.97)


def test_reverb_tolerant_true_when_all_clear_bar():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.95, 0.0, 50), (0.3, 0.9, 0.4, 50), (0.6, 0.85, 0.6, 50)),
        accuracy_bar=BAR)
    assert out.reverb_tolerant is True


def test_reverb_tolerant_false_when_one_fails():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.95, 0.0, 50), (1.0, 0.6, 0.8, 50)), accuracy_bar=BAR)
    assert out.reverb_tolerant is False


def test_reliable_ceiling_is_highest_contiguous_rt60():
    from model.reverb_robustness import reverb_robustness

    # reliable from dry up through 0.6, breaks at 1.0.
    out = reverb_robustness(
        _pts((0.0, 0.95, 0.0, 50), (0.3, 0.9, 0.4, 50),
             (0.6, 0.85, 0.6, 50), (1.0, 0.6, 0.8, 50)),
        accuracy_bar=BAR)
    assert out.reliable_ceiling_s == pytest.approx(0.6)


def test_reliable_ceiling_none_when_dry_fails():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.5, 0.0, 50), (0.3, 0.9, 0.4, 50)), accuracy_bar=BAR)
    assert out.reliable_ceiling_s is None


def test_worst_accuracy_and_rt60():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.95, 0.0, 50), (1.0, 0.6, 0.8, 50)), accuracy_bar=BAR)
    assert out.worst_accuracy == pytest.approx(0.6)
    assert out.worst_rt60_s == pytest.approx(1.0)


def test_no_dry_point_has_none_clean_accuracy():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(_pts((0.3, 0.9, 0.4, 50)), accuracy_bar=BAR)
    assert out.clean_accuracy is None
    assert out.reliable_ceiling_s is None


def test_empty_rejected():
    from model.reverb_robustness import reverb_robustness

    with pytest.raises(ValueError):
        reverb_robustness([], accuracy_bar=BAR)


def test_rt60_out_of_range_rejected():
    from model.reverb_robustness import reverb_robustness

    with pytest.raises(ValueError):
        reverb_robustness(_pts((-0.1, 0.9, 0.0, 50)), accuracy_bar=BAR)


def test_accuracy_out_of_range_rejected():
    from model.reverb_robustness import reverb_robustness

    with pytest.raises(ValueError):
        reverb_robustness(_pts((0.0, 1.5, 0.0, 50)), accuracy_bar=BAR)


def test_tail_fraction_out_of_range_rejected():
    from model.reverb_robustness import reverb_robustness

    with pytest.raises(ValueError):
        reverb_robustness(_pts((0.0, 0.9, 1.2, 50)), accuracy_bar=BAR)


def test_nonpositive_n_rejected():
    from model.reverb_robustness import reverb_robustness

    with pytest.raises(ValueError):
        reverb_robustness(_pts((0.0, 0.9, 0.0, 0)), accuracy_bar=BAR)


def test_verdict_nonempty():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.95, 0.0, 50), (0.6, 0.88, 0.6, 50)), accuracy_bar=BAR)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.reverb_robustness import reverb_robustness

    out = reverb_robustness(
        _pts((0.0, 0.95, 0.0, 50), (0.6, 0.88, 0.6, 50)), accuracy_bar=BAR)
    back = json.loads(json.dumps(out.to_dict()))
    assert "reverb_tolerant" in back
    assert "reliable_ceiling_s" in back
    assert "clean_accuracy" in back
    assert len(back["points"]) == 2
    assert "tail_fraction" in back["points"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "reverb_robustness.json" in names


# --- heavy build path (trains + reverberates eval sets) --------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.reverb_robustness import build_reverb_robustness

    out = build_reverb_robustness(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=16,
        rt60s=(0.0, 0.6), eval_snr=None,
        out_dir=tmp_path,
    )
    assert len(out.points) == 2
    assert out.clean_accuracy is not None
    assert (tmp_path / "reverb_robustness.json").is_file()
    assert (tmp_path / "reverb_robustness.md").is_file()
