"""Host mirror of the calming-intervention policy (model/intervention.py).

M9 (the on-device calming cue) is device-side, but its *decision* — when does a
sustained-stress reading earn a calming prompt, and how do we avoid nagging — is
pure logic that belongs in host-testable code, exactly as ``model.pipeline``
mirrors the Kotlin stream pipeline. The policy consumes the detector's per-window
:class:`StressState` and decides whether to fire a single calming cue. It must:

  * fire only after stress is *sustained* (N consecutive voiced+stressed windows),
    so a one-window blip never triggers it;
  * fire at most once per episode and then rate-limit (cooldown) so it never nags;
  * be dismissible — a user dismissal also opens the cooldown;
  * reset its streak on any calm / unvoiced window.

These tests pin those four behaviors plus reset. Host-only; no device, no token.
"""

from __future__ import annotations

from model.detector import StressState
from model.intervention import InterventionConfig, InterventionPolicy


def _voiced(stressed: bool) -> StressState:
    s = 0.8 if stressed else 0.2
    return StressState(voiced=True, raw_score=s, level=s, stressed=stressed)


_UNVOICED = StressState(voiced=False, raw_score=None, level=None, stressed=False)


def _fires(policy: InterventionPolicy, state: StressState) -> bool:
    return policy.update(state).fire


def test_does_not_fire_before_sustained():
    p = InterventionPolicy(InterventionConfig(sustain_windows=3, cooldown_windows=10))
    assert _fires(p, _voiced(True)) is False
    assert _fires(p, _voiced(True)) is False  # only 2 of 3


def test_fires_once_after_sustained_stress():
    p = InterventionPolicy(InterventionConfig(sustain_windows=3, cooldown_windows=10))
    assert _fires(p, _voiced(True)) is False
    assert _fires(p, _voiced(True)) is False
    assert _fires(p, _voiced(True)) is True   # 3rd consecutive -> fire
    assert _fires(p, _voiced(True)) is False  # stays stressed but must not nag


def test_calm_window_resets_the_streak():
    p = InterventionPolicy(InterventionConfig(sustain_windows=3, cooldown_windows=10))
    _fires(p, _voiced(True))
    _fires(p, _voiced(True))
    assert _fires(p, _voiced(False)) is False   # streak broken
    assert _fires(p, _voiced(True)) is False     # count restarts at 1
    assert _fires(p, _voiced(True)) is False     # 2
    assert _fires(p, _voiced(True)) is True      # 3 -> fire


def test_unvoiced_window_resets_the_streak():
    p = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=5))
    assert _fires(p, _voiced(True)) is False
    assert _fires(p, _UNVOICED) is False         # silence breaks sustain
    assert _fires(p, _voiced(True)) is False      # restart at 1
    assert _fires(p, _voiced(True)) is True       # 2 -> fire


def test_rate_limited_during_cooldown():
    p = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=3))
    assert _fires(p, _voiced(True)) is False
    assert _fires(p, _voiced(True)) is True       # fire
    # stays maximally stressed through the cooldown -> never refires
    for _ in range(3):
        assert _fires(p, _voiced(True)) is False
    assert p.in_cooldown is False                  # cooldown elapsed


def test_refires_after_cooldown_and_new_sustain():
    p = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=2))
    assert _fires(p, _voiced(True)) is False
    assert _fires(p, _voiced(True)) is True        # first fire
    _fires(p, _voiced(True))                        # cooldown tick 1
    _fires(p, _voiced(True))                        # cooldown tick 2 (elapses)
    # new sustained episode fires again
    fired = _fires(p, _voiced(True)) or _fires(p, _voiced(True))
    assert fired is True


def test_dismiss_opens_cooldown_and_prevents_immediate_refire():
    p = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=4))
    _fires(p, _voiced(True))
    assert _fires(p, _voiced(True)) is True
    p.dismiss()
    assert p.in_cooldown is True
    assert _fires(p, _voiced(True)) is False


def test_reset_clears_state():
    p = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=4))
    _fires(p, _voiced(True))
    _fires(p, _voiced(True))
    p.reset()
    assert p.in_cooldown is False
    assert _fires(p, _voiced(True)) is False        # streak restarted
    assert _fires(p, _voiced(True)) is True
