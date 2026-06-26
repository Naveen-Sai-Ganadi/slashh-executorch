"""Detector onset/offset latency A/B (TDD, red first).

Every detector artifact so far measures per-window *accuracy* (robustness,
tuning, flicker). None measures *time-to-alarm*: the EMA smoothing (alpha 0.4)
plus dual-threshold hysteresis (enter 0.6 / release 0.45) deliberately delay the
latch so it can't flicker — but that same smoothing means the alarm appears
several hop-windows after stress actually begins, and lingers after it ends.
With HOP_SECONDS = 1.0 each window is a second of wall-clock, so this delay is
directly the product-visible "how long until it notices / lets go" number.

This A/B drives the real StressDetector through synthetic calm->stressed->calm
episodes and reports the onset latency (windows/seconds from true stress onset to
the latch firing), the detection rate (did it fire at all), and the release
latency after stress ends. A responsive detector fires within a small budget on
every episode.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested with
hand-set latencies; the heavy build path trains one tiny net and runs episodes.
"""

from __future__ import annotations

import json

import pytest

HOP = 1.0


def _episodes(*rows):
    """Build (onset_latency_windows|None, release_latency_windows|None) records."""
    return list(rows)


# --- pure reduction logic (no training) ------------------------------------

def test_episode_seconds_conversion():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((3, 2)), hop_seconds=HOP)
    ep = out.episodes[0]
    assert ep.detected is True
    assert ep.released is True
    assert ep.onset_latency_s == pytest.approx(3 * HOP)
    assert ep.release_latency_s == pytest.approx(2 * HOP)


def test_detection_rate():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((2, 1), (None, None), (4, 3)), hop_seconds=HOP)
    assert out.detection_rate == pytest.approx(2 / 3)


def test_median_onset_over_detected_only():
    from model.detection_latency import detection_latency

    # detected onsets 2 and 4 -> median 3 windows; the miss is excluded.
    out = detection_latency(_episodes((2, 1), (None, None), (4, 3)), hop_seconds=HOP)
    assert out.median_onset_windows == pytest.approx(3.0)
    assert out.median_onset_s == pytest.approx(3.0 * HOP)


def test_max_onset_seconds():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((2, 1), (5, 2)), hop_seconds=HOP)
    assert out.max_onset_s == pytest.approx(5 * HOP)


def test_release_rate_and_median_among_detected():
    from model.detection_latency import detection_latency

    # detected episodes: (2,1) released, (3,None) stuck -> release_rate 1/2.
    out = detection_latency(_episodes((2, 1), (3, None)), hop_seconds=HOP)
    assert out.release_rate == pytest.approx(0.5)
    assert out.median_release_s == pytest.approx(1.0 * HOP)


def test_responsive_when_fast_and_complete():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((1, 1), (2, 1)), hop_seconds=HOP, onset_budget_s=3.0)
    assert out.responsive is True


def test_not_responsive_when_misses():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((1, 1), (None, None)), hop_seconds=HOP, onset_budget_s=3.0)
    assert out.responsive is False


def test_not_responsive_when_too_slow():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((6, 1), (7, 1)), hop_seconds=HOP, onset_budget_s=3.0)
    assert out.responsive is False


def test_no_detections_has_none_medians():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((None, None), (None, None)), hop_seconds=HOP)
    assert out.detection_rate == pytest.approx(0.0)
    assert out.median_onset_s is None
    assert out.responsive is False


def test_empty_rejected():
    from model.detection_latency import detection_latency

    with pytest.raises(ValueError):
        detection_latency([], hop_seconds=HOP)


def test_nonpositive_hop_rejected():
    from model.detection_latency import detection_latency

    with pytest.raises(ValueError):
        detection_latency(_episodes((2, 1)), hop_seconds=0.0)


def test_negative_latency_rejected():
    from model.detection_latency import detection_latency

    with pytest.raises(ValueError):
        detection_latency(_episodes((-1, 1)), hop_seconds=HOP)


def test_verdict_nonempty():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((2, 1), (3, 2)), hop_seconds=HOP)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.detection_latency import detection_latency

    out = detection_latency(_episodes((2, 1), (None, None)), hop_seconds=HOP)
    back = json.loads(json.dumps(out.to_dict()))
    assert "detection_rate" in back
    assert "median_onset_s" in back
    assert "responsive" in back
    assert len(back["episodes"]) == 2
    assert "onset_latency_s" in back["episodes"][0]


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "detection_latency.json" in names


# --- heavy build path (trains + drives real episodes) ----------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.detection_latency import build_detection_latency

    out = build_detection_latency(
        seed=0, epochs=2, n_per_class=8,
        snr_levels=(None, 0.0), n_episodes=2,
        calm_windows=3, stress_windows=8, tail_windows=6,
        out_dir=tmp_path,
    )
    assert len(out.episodes) == 4  # 2 SNRs x 2 episodes
    assert 0.0 <= out.detection_rate <= 1.0
    assert (tmp_path / "detection_latency.json").is_file()
    assert (tmp_path / "detection_latency.md").is_file()
