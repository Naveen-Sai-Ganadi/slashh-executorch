"""Non-gating confidence telemetry over the detector's state (model/telemetry.py).

``model.confidence`` turns a raw score into a calibrated probability + a 0.5..1
confidence meter. ``model.telemetry`` attaches that read-out to a ``StressState``
WITHOUT changing the decision: the ``stressed`` flag is copied verbatim from the
frozen detector gate (which latches on the *raw* score at ``STRESS_THRESHOLD``).
These tests pin that the read-out (a) mirrors the gate decision unchanged, (b)
exposes a calibrated confidence in [0.5, 1] matching ``model.confidence``, and
(c) is ``None`` for a stateless/unvoiced window (no score to calibrate). It also
checks the ``HostStressPipeline`` convenience property that surfaces the read-out
for the most recent window. Host-only; no device, no token.
"""

from __future__ import annotations

import torch

from model.confidence import DEFAULT_TEMPERATURE, confidence
from model.detector import StressState
from model.telemetry import ConfidenceReadout, readout


def test_readout_is_none_for_unscored_state():
    neutral = StressState(voiced=False, raw_score=None, level=None, stressed=False)
    assert readout(neutral) is None


def test_readout_mirrors_gate_decision_unchanged():
    # The decision comes from the frozen gate; telemetry must not alter it.
    for stressed in (False, True):
        st = StressState(voiced=True, raw_score=0.58, level=0.58, stressed=stressed)
        r = readout(st)
        assert isinstance(r, ConfidenceReadout)
        assert r.stressed is stressed
        assert r.raw_score == 0.58


def test_readout_confidence_matches_confidence_module():
    st = StressState(voiced=True, raw_score=0.7, level=0.7, stressed=True)
    r = readout(st)
    assert abs(r.confidence - confidence(0.7)) < 1e-9
    assert 0.5 <= r.confidence <= 1.0


def test_readout_calibrated_probability_in_unit_interval():
    for raw in (0.02, 0.3, 0.45, 0.6, 0.97):
        st = StressState(voiced=True, raw_score=raw, level=raw, stressed=raw >= 0.6)
        r = readout(st)
        assert 0.0 <= r.calibrated_probability <= 1.0


def test_readout_honors_temperature_argument():
    st = StressState(voiced=True, raw_score=0.7, level=0.7, stressed=True)
    # T=1 is identity: calibrated probability equals the raw score.
    r1 = readout(st, temperature=1.0)
    assert abs(r1.calibrated_probability - 0.7) < 1e-6
    # default sub-unit temperature sharpens an above-0.5 score upward.
    rd = readout(st, temperature=DEFAULT_TEMPERATURE)
    assert rd.calibrated_probability > 0.7


def test_pipeline_exposes_last_readout():
    from model.model import StressNet
    from model.pipeline import HostStressPipeline

    pipe = HostStressPipeline(StressNet())
    # before any window, neutral state -> no read-out
    assert pipe.last_confidence is None
    pipe.process_window(torch.zeros(48000), voiced=True)
    r = pipe.last_confidence
    assert isinstance(r, ConfidenceReadout)
    assert 0.5 <= r.confidence <= 1.0
    # the read-out's decision matches the pipeline's gated state, unchanged.
    assert r.stressed is pipe.state.stressed
