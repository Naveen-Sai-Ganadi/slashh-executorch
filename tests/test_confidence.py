"""Tests for the non-gating calibrated-confidence read-out (model/confidence.py).

The score-calibration analysis found StressNet is *under-confident*: raw scores
hedge toward 0.5 even when right, and a temperature ``T≈0.41`` sharpens them into
well-calibrated probabilities. ``model.confidence`` turns that into a reusable
read-out for a UI/telemetry confidence meter, WITHOUT touching the frozen
detector gate. These tests pin the transform's identities (``T=1`` identity,
0.5 fixed point), the direction of the correction (``T<1`` sharpens, ``T>1``
softens), the symmetric [0.5, 1] confidence, type preservation, and exact
numerical agreement with the temperature transform inside ``model.calibration``.
"""

from __future__ import annotations

import torch

from model.confidence import (
    DEFAULT_TEMPERATURE,
    calibrated_probability,
    confidence,
    temperature_scale,
)


def test_unit_temperature_is_identity():
    for p in (0.01, 0.2, 0.5, 0.6, 0.99):
        assert abs(temperature_scale(p, 1.0) - p) < 1e-6


def test_half_is_a_fixed_point():
    for T in (0.41, 1.0, 2.5):
        assert abs(temperature_scale(0.5, T) - 0.5) < 1e-9


def test_sub_unit_temperature_sharpens_toward_extremes():
    # T < 1 corrects under-confidence: push probabilities AWAY from 0.5.
    assert calibrated_probability(0.6, 0.41) > 0.6
    assert calibrated_probability(0.4, 0.41) < 0.4


def test_super_unit_temperature_softens_toward_half():
    # T > 1 pulls probabilities TOWARD 0.5.
    assert 0.5 < temperature_scale(0.9, 2.0) < 0.9
    assert 0.1 < temperature_scale(0.1, 2.0) < 0.5


def test_confidence_is_symmetric_and_in_upper_half():
    for p in (0.05, 0.3, 0.5, 0.7, 0.95):
        c = confidence(p)
        assert 0.5 <= c <= 1.0
        assert abs(confidence(p) - confidence(1.0 - p)) < 1e-6


def test_type_preservation_float_and_tensor():
    assert isinstance(temperature_scale(0.6, 0.41), float)
    t = temperature_scale(torch.tensor([0.2, 0.8]), 0.41)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2,)


def test_extremes_stay_finite():
    for p in (0.0, 1.0):
        v = temperature_scale(p, 0.41)
        assert 0.0 <= v <= 1.0 and v == v  # finite, in range


def test_matches_calibration_transform_exactly():
    # confidence.temperature_scale must equal the recover-logit/scale/re-sigmoid
    # transform used inside model.calibration (single source of truth).
    scores = torch.tensor([0.02, 0.3, 0.5, 0.57, 0.9, 0.999])
    T = 0.41
    eps = 1e-6
    p = scores.clamp(eps, 1.0 - eps)
    expected = torch.sigmoid(torch.log(p / (1.0 - p)) / T)
    got = temperature_scale(scores, T)
    assert torch.allclose(got, expected, atol=1e-7)


def test_default_temperature_matches_calibration_finding():
    # The shipped finding is sub-unit (under-confident model).
    assert 0.0 < DEFAULT_TEMPERATURE < 1.0
