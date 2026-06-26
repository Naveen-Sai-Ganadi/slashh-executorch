"""Host reference for the on-device stress decision loop.

This is the Python golden spec of ``StressPipeline`` (android/.../
StressPipeline.kt): it turns a stream of per-window classifier scores into a
steady, non-flickering ``stressed`` signal via EMA smoothing and dual-threshold
hysteresis. The Android implementation must agree with this window-for-window.

Keeping the decision logic here — decoupled from the model and the VAD, which
already have Python equivalents — makes it unit-testable and **tunable** on the
host (sweep thresholds / alpha against labelled score traces) without a device.

    from model.detector import StressDetector
    d = StressDetector()
    for raw, voiced in stream:          # raw: model score, voiced: VAD gate
        state = d.update(raw, voiced)
        meter, latched = state.level, state.stressed

Pure and synchronous; owns no threads and no audio. See model/audio_config.py
for the frozen threshold/alpha defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio_config import EMA_ALPHA, RELEASE_THRESHOLD, STRESS_THRESHOLD


@dataclass(frozen=True)
class StressState:
    """One window's decision output (mirror of the Kotlin ``StressState``)."""

    voiced: bool
    raw_score: float | None   # raw model score, or None if the window was gated
    level: float | None       # EMA-smoothed score, or None before any voiced window
    stressed: bool            # hysteresis latch — True while in the stressed band


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class StressDetector:
    """EMA + dual-threshold hysteresis over per-window classifier scores.

    Args mirror the device knobs in :mod:`model.audio_config`:

    - ``stress_threshold``: EMA level at/above which we enter "stressed".
    - ``release_threshold``: EMA level below which we leave it. Strictly less
      than ``stress_threshold`` so the latch can't flicker at the boundary.
    - ``ema_alpha``: weight on the newest window (1.0 = no smoothing).
    """

    def __init__(
        self,
        *,
        stress_threshold: float = STRESS_THRESHOLD,
        release_threshold: float = RELEASE_THRESHOLD,
        ema_alpha: float = EMA_ALPHA,
    ) -> None:
        if not (0.0 <= release_threshold < stress_threshold <= 1.0):
            raise ValueError(
                "require 0 <= release_threshold < stress_threshold <= 1 "
                f"(got release={release_threshold}, stress={stress_threshold})"
            )
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self.stress_threshold = stress_threshold
        self.release_threshold = release_threshold
        self.ema_alpha = ema_alpha
        self._ema: float | None = None     # None until the first voiced window
        self._stressed = False
        self._voiced_seen = False

    def update(self, raw_score: float | None, voiced: bool) -> StressState:
        """Process one window. Mirrors ``StressPipeline.onWindow``.

        On an unvoiced window the model is not run: the meter holds the last EMA
        (or None before any voiced window) and the latch is unchanged.
        """
        if not voiced:
            return StressState(
                voiced=False,
                raw_score=None,
                level=self._ema if self._voiced_seen else None,
                stressed=self._stressed,
            )

        self._voiced_seen = True
        raw = _clamp01(float(raw_score))

        # EMA: seed to the first raw score, then exponentially smooth.
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._ema

        # Hysteresis: separate enter/exit thresholds; hold the latch in between.
        if self._ema >= self.stress_threshold:
            self._stressed = True
        elif self._ema < self.release_threshold:
            self._stressed = False

        return StressState(
            voiced=True, raw_score=raw, level=self._ema, stressed=self._stressed
        )

    def run(self, scores, voiced) -> list[StressState]:
        """Process a sequence of (score, voiced) pairs, returning each state."""
        return [self.update(s, v) for s, v in zip(scores, voiced)]

    def reset(self) -> None:
        """Clear smoothing/latch state (e.g. after backgrounding the screen)."""
        self._ema = None
        self._stressed = False
        self._voiced_seen = False
