"""Tests for the host reference stress detector (model/detector.py).

This Python detector is the **golden spec** for the on-device decision loop in
android/.../StressPipeline.kt: raw classifier score + voiced flag in, smoothed
level + hysteresis latch out. Mirroring it on the host makes the EMA + dual-
threshold hysteresis behaviour testable and tunable without a device. The two
implementations must agree window-for-window.

Reference Kotlin semantics (StressPipeline.onWindow):
  - unvoiced window: model not run; level = held EMA (or None before any
    voiced window); latch unchanged.
  - first voiced window: EMA initialised to the raw score (no smoothing).
  - later voiced: ema = alpha*raw + (1-alpha)*ema.
  - hysteresis: enter at STRESS_THRESHOLD, leave only below RELEASE_THRESHOLD.
  - raw score coerced into [0, 1].
"""

from __future__ import annotations

import math

from model.audio_config import EMA_ALPHA, RELEASE_THRESHOLD, STRESS_THRESHOLD
from model.detector import StressDetector, StressState


def test_defaults_match_audio_config() -> None:
    d = StressDetector()
    assert d.stress_threshold == STRESS_THRESHOLD
    assert d.release_threshold == RELEASE_THRESHOLD
    assert d.ema_alpha == EMA_ALPHA


def test_unvoiced_before_any_voice_reports_nothing() -> None:
    d = StressDetector()
    s = d.update(raw_score=0.9, voiced=False)
    assert isinstance(s, StressState)
    assert s.voiced is False
    assert s.raw_score is None   # model not run on unvoiced windows
    assert s.level is None       # no EMA established yet
    assert s.stressed is False


def test_first_voiced_initialises_ema_to_raw() -> None:
    d = StressDetector()
    s = d.update(raw_score=0.7, voiced=True)
    assert s.voiced is True
    assert s.raw_score == 0.7
    assert s.level == 0.7        # EMA seeded to the raw score, no smoothing
    assert s.stressed is True    # 0.7 >= STRESS_THRESHOLD (0.6)


def test_ema_smoothing_formula() -> None:
    d = StressDetector(ema_alpha=0.4)
    d.update(raw_score=0.8, voiced=True)            # ema = 0.8
    s = d.update(raw_score=0.3, voiced=True)        # ema = 0.4*0.3 + 0.6*0.8
    assert math.isclose(s.level, 0.4 * 0.3 + 0.6 * 0.8, rel_tol=1e-6)


def test_raw_score_is_clamped_to_unit_interval() -> None:
    d = StressDetector()
    hi = d.update(raw_score=1.5, voiced=True)
    assert hi.raw_score == 1.0 and hi.level == 1.0
    d.reset()
    lo = d.update(raw_score=-0.3, voiced=True)
    assert lo.raw_score == 0.0 and lo.level == 0.0


def test_hysteresis_latches_between_thresholds() -> None:
    # custom thresholds for a clean band: enter 0.6, leave below 0.4
    d = StressDetector(stress_threshold=0.6, release_threshold=0.4, ema_alpha=1.0)
    assert d.update(0.65, voiced=True).stressed is True    # cross enter -> latched
    assert d.update(0.50, voiced=True).stressed is True    # in band -> stays latched
    assert d.update(0.45, voiced=True).stressed is True    # still >= release
    assert d.update(0.39, voiced=True).stressed is False   # below release -> off
    assert d.update(0.50, voiced=True).stressed is False   # in band again -> stays off


def test_unvoiced_after_voice_holds_level_and_latch() -> None:
    d = StressDetector(ema_alpha=1.0)
    d.update(0.8, voiced=True)                      # level 0.8, stressed True
    s = d.update(raw_score=0.1, voiced=False)       # silence: model not run
    assert s.voiced is False
    assert s.raw_score is None
    assert s.level == 0.8        # holds the last EMA
    assert s.stressed is True    # latch held through silence


def test_reset_clears_all_state() -> None:
    d = StressDetector()
    d.update(0.9, voiced=True)
    d.reset()
    s = d.update(0.1, voiced=False)
    assert s.level is None and s.stressed is False


def test_run_processes_a_sequence() -> None:
    d = StressDetector(stress_threshold=0.6, release_threshold=0.45, ema_alpha=1.0)
    scores = [0.1, 0.7, 0.5, 0.2]
    voiced = [True, True, True, True]
    states = d.run(scores, voiced)
    assert len(states) == 4
    assert [s.stressed for s in states] == [False, True, True, False]
    assert [round(s.level, 3) for s in states] == [0.1, 0.7, 0.5, 0.2]
