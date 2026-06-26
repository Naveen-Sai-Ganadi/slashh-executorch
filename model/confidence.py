"""Calibrated confidence from a raw StressNet score — temperature scaling.

The score-calibration analysis (:mod:`model.calibration`, recorded in
``docs/benchmarks/calibration.json``) found the shipped model is
**under-confident**: its ``sigmoid`` scores hedge toward 0.5 even when right
(ECE 0.242), and a single temperature ``T ≈ 0.41`` sharpens them into
well-calibrated probabilities (ECE → 0.102). This module turns that finding
into a small, reusable **read-out**: a calibrated probability and a 0.5..1
"confidence in the decision" suitable for a UI meter or telemetry.

It is deliberately **non-gating**. The detector gate stays on the *raw* score at
the frozen ``STRESS_THRESHOLD`` — the detector-knob A/B showed that lowering the
effective gate trips false alarms in the worst-case noise tail, so recalibration
must not feed back into the latch. Use this only to *display* how confident a
decision is, never to make it.

    from model.confidence import calibrated_probability, confidence
    p = calibrated_probability(raw_score)   # well-calibrated probability
    meter = confidence(raw_score)           # 0.5 (unsure) .. 1.0 (certain)

The transform matches the recover-logit / scale / re-sigmoid used inside
:mod:`model.calibration` exactly. Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import math

import torch

# Sub-unit temperature from the calibration finding (under-confident model).
# Source: docs/benchmarks/calibration.json -> "temperature".
DEFAULT_TEMPERATURE = 0.41

_EPS = 1e-6  # matches model.calibration._fit_temperature

__all__ = [
    "DEFAULT_TEMPERATURE",
    "temperature_scale",
    "calibrated_probability",
    "confidence",
]


def temperature_scale(score, temperature: float = DEFAULT_TEMPERATURE):
    """Recalibrate a ``sigmoid`` probability ``score`` by temperature ``T``.

    Recovers the logit from the (clamped) probability, divides it by ``T``, and
    re-applies the sigmoid — the standard temperature-scaling transform. ``T<1``
    sharpens (corrects under-confidence, pushing away from 0.5); ``T>1`` softens;
    ``T=1`` is the identity. 0.5 is a fixed point for every ``T``.

    Accepts a Python ``float`` or a :class:`torch.Tensor` and returns the same
    type. ``temperature`` must be > 0.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if isinstance(score, torch.Tensor):
        p = score.clamp(_EPS, 1.0 - _EPS)
        logit = torch.log(p / (1.0 - p))
        return torch.sigmoid(logit / temperature)

    p = min(max(float(score), _EPS), 1.0 - _EPS)
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def calibrated_probability(score, temperature: float = DEFAULT_TEMPERATURE):
    """The temperature-recalibrated probability of stress (alias for clarity)."""
    return temperature_scale(score, temperature)


def confidence(score, temperature: float = DEFAULT_TEMPERATURE):
    """Confidence *in the decision*, in ``[0.5, 1]``, on the calibrated score.

    0.5 means maximally unsure (calibrated probability at the midpoint); 1.0
    means certain. Symmetric: a calibrated 0.2 and 0.8 carry equal confidence
    (0.8) in opposite directions. For a tensor input, returns a tensor.
    """
    p = calibrated_probability(score, temperature)
    if isinstance(p, torch.Tensor):
        return torch.maximum(p, 1.0 - p)
    return max(p, 1.0 - p)
