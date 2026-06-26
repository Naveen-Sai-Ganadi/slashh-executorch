"""Detector silence-hold / gap robustness A/B (TDD, red first).

Every existing detector artifact (tuning, robustness, noise A/B, onset/offset
latency) drives the detector with a *continuous* voiced stream (voiced=True every
window). None exercises the VAD-gated path: on an unvoiced window the detector
freezes its EMA and holds the latch (model/detector.py update(), voiced=False
branch). That gating has a product-visible consequence — if the user goes silent
right after a stress episode ends, the alarm stays latched for the *entire*
silence because the EMA can only decay on voiced windows. With HOP_SECONDS = 1.0
each gap window is a second the alarm lingers.

This A/B drives the real StressDetector through calm->stress(latch on)->SILENCE
GAP->calm episodes and measures, per episode, how long after the true stress
offset the latch clears, counted two ways: in *voiced* windows (the EMA decay
budget, gap-independent) and in *wall-clock* windows (voiced + the unvoiced gap).
The difference is the silence inflation — seconds the alarm persists purely
because the user fell quiet. A latch that "holds through silence" inflates by
exactly the gap; one that drops during silence would not.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set records; the heavy build trains one tiny net and runs real episodes.
"""

from __future__ import annotations

import json

import pytest

HOP = 1.0


def _episodes(*rows):
    """Build (gap_windows, voiced_release_windows|None, wallclock_release_windows|None)."""
    return list(rows)


# --- pure reduction logic (no training) ------------------------------------

def test_episode_seconds_and_inflation():
    from model.silence_hold import silence_hold

    # gap 3, releases after 2 voiced windows, 5 wall-clock windows (3 gap + 2).
    out = silence_hold(_episodes((3, 2, 5)), hop_seconds=HOP)
    ep = out.episodes[0]
    assert ep.released is True
    assert ep.voiced_release_s == pytest.approx(2 * HOP)
    assert ep.wallclock_release_s == pytest.approx(5 * HOP)
    assert ep.inflation_windows == 3
    assert ep.inflation_s == pytest.approx(3 * HOP)


def test_release_rate_over_all_episodes():
    from model.silence_hold import silence_hold

    out = silence_hold(_episodes((0, 2, 2), (3, None, None), (10, 2, 12)), hop_seconds=HOP)
    assert out.release_rate == pytest.approx(2 / 3)


def test_median_and_max_wallclock_over_released():
    from model.silence_hold import silence_hold

    # released wall-clock windows: 2 and 12 -> median 7, max 12.
    out = silence_hold(_episodes((0, 2, 2), (3, None, None), (10, 2, 12)), hop_seconds=HOP)
    assert out.median_wallclock_release_s == pytest.approx(7.0 * HOP)
    assert out.max_wallclock_release_s == pytest.approx(12.0 * HOP)


def test_median_voiced_release_gap_independent():
    from model.silence_hold import silence_hold

    # both released episodes need 2 voiced windows regardless of gap size.
    out = silence_hold(_episodes((0, 2, 2), (10, 2, 12)), hop_seconds=HOP)
    assert out.median_voiced_release_s == pytest.approx(2.0 * HOP)


def test_max_inflation_seconds():
    from model.silence_hold import silence_hold

    out = silence_hold(_episodes((0, 2, 2), (10, 2, 12)), hop_seconds=HOP)
    # inflations: 0 and 10 -> max 10.
    assert out.max_inflation_s == pytest.approx(10.0 * HOP)


def test_holds_through_silence_true_when_inflation_matches_gap():
    from model.silence_hold import silence_hold

    # every gapped+released episode inflates by exactly its gap -> holds.
    out = silence_hold(_episodes((0, 2, 2), (3, 2, 5), (10, 2, 12)), hop_seconds=HOP)
    assert out.holds_through_silence is True


def test_holds_through_silence_false_when_latch_drops_in_gap():
    from model.silence_hold import silence_hold

    # gap 10 but inflation only 4 -> latch dropped mid-silence, didn't hold.
    out = silence_hold(_episodes((10, 2, 6)), hop_seconds=HOP)
    assert out.holds_through_silence is False


def test_clears_within_budget():
    from model.silence_hold import silence_hold

    fast = silence_hold(_episodes((0, 2, 2), (3, 2, 5)), hop_seconds=HOP, clear_budget_s=8.0)
    assert fast.clears_within_budget is True
    slow = silence_hold(_episodes((10, 2, 12)), hop_seconds=HOP, clear_budget_s=8.0)
    assert slow.clears_within_budget is False


def test_no_release_has_none_medians_and_zero_rate():
    from model.silence_hold import silence_hold

    out = silence_hold(_episodes((3, None, None), (5, None, None)), hop_seconds=HOP)
    assert out.release_rate == pytest.approx(0.0)
    assert out.median_wallclock_release_s is None
    assert out.median_voiced_release_s is None
    assert out.clears_within_budget is False


def test_empty_rejected():
    from model.silence_hold import silence_hold

    with pytest.raises(ValueError):
        silence_hold([], hop_seconds=HOP)


def test_nonpositive_hop_rejected():
    from model.silence_hold import silence_hold

    with pytest.raises(ValueError):
        silence_hold(_episodes((3, 2, 5)), hop_seconds=0.0)


def test_negative_gap_rejected():
    from model.silence_hold import silence_hold

    with pytest.raises(ValueError):
        silence_hold(_episodes((-1, 2, 5)), hop_seconds=HOP)


def test_wallclock_below_voiced_rejected():
    from model.silence_hold import silence_hold

    # wall-clock release cannot be fewer windows than voiced-only release.
    with pytest.raises(ValueError):
        silence_hold(_episodes((3, 5, 2)), hop_seconds=HOP)


def test_inconsistent_none_rejected():
    from model.silence_hold import silence_hold

    # one of voiced/wallclock None but not the other is inconsistent.
    with pytest.raises(ValueError):
        silence_hold(_episodes((3, 2, None)), hop_seconds=HOP)


def test_verdict_nonempty():
    from model.silence_hold import silence_hold

    out = silence_hold(_episodes((0, 2, 2), (3, 2, 5)), hop_seconds=HOP)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.silence_hold import silence_hold

    out = silence_hold(_episodes((3, 2, 5), (10, None, None)), hop_seconds=HOP)
    back = json.loads(json.dumps(out.to_dict()))
    assert "release_rate" in back
    assert "holds_through_silence" in back
    assert "max_inflation_s" in back
    assert len(back["episodes"]) == 2
    assert "inflation_s" in back["episodes"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "silence_hold.json" in names


# --- heavy build path (trains + drives real gap episodes) ------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.silence_hold import build_silence_hold

    out = build_silence_hold(
        seed=0, epochs=2, n_per_class=8,
        snr_levels=(None, 0.0), gap_windows_list=(0, 4),
        calm_pre=3, stress_windows=8, tail_windows=12,
        out_dir=tmp_path,
    )
    assert len(out.episodes) == 4  # 2 SNRs x 2 gap sizes
    assert 0.0 <= out.release_rate <= 1.0
    assert (tmp_path / "silence_hold.json").is_file()
    assert (tmp_path / "silence_hold.md").is_file()
