"""Non-gating confidence telemetry over a detector :class:`StressState`.

The detector decides — it latches ``stressed`` on the *raw* model score at the
frozen ``STRESS_THRESHOLD`` (the detector-knob A/B showed that moving the gate
trips false alarms in the noise tail). This module does NOT touch that decision.
It only *annotates* a finished state with the calibrated read-out from
:mod:`model.confidence`, so a UI/telemetry layer can show "stressed — 0.87
confident" without ever influencing whether the latch fired.

    from model.telemetry import readout
    r = readout(state)            # None for an unscored (unvoiced/neutral) window
    r.stressed                    # the gate's decision, copied verbatim
    r.calibrated_probability      # temperature-corrected probability of stress
    r.confidence                  # 0.5 (unsure) .. 1.0 (certain), symmetric

Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

from dataclasses import dataclass

from .confidence import DEFAULT_TEMPERATURE, calibrated_probability, confidence
from .detector import StressState


@dataclass(frozen=True)
class ConfidenceReadout:
    """A calibrated, non-gating view of one decided window.

    ``stressed`` is the detector's decision, unchanged; the rest is derived from
    the raw score purely for display/telemetry.
    """

    raw_score: float
    stressed: bool
    calibrated_probability: float
    confidence: float


def readout(
    state: StressState, *, temperature: float = DEFAULT_TEMPERATURE
) -> ConfidenceReadout | None:
    """Annotate a decided ``state`` with calibrated confidence — or ``None``.

    Returns ``None`` when the window carries no score (unvoiced / neutral /
    freshly reset), since there is nothing to calibrate. Otherwise the raw score
    is mapped through :mod:`model.confidence`; the gate's ``stressed`` decision is
    carried through untouched.
    """
    if state.raw_score is None:
        return None
    raw = float(state.raw_score)
    return ConfidenceReadout(
        raw_score=raw,
        stressed=bool(state.stressed),
        calibrated_probability=float(calibrated_probability(raw, temperature)),
        confidence=float(confidence(raw, temperature)),
    )
