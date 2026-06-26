"""End-to-end host session (model/session.py) — the full on-device loop, host-side.

Three host components now exist in isolation: the pipeline (waveform -> score ->
gate -> StressState), the calibrated confidence read-out, and the calming-
intervention policy. ``HostStressSession`` composes them so one call per window
yields the whole outcome — state + confidence + intervention decision — mirroring
what the device loop does each hop. These tests pin the bundling, that an
unvoiced window has no confidence and never fires, that sustained stress
eventually fires the cue, and that dismiss()/reset() reach the policy. Host-only.
"""

from __future__ import annotations

import torch
from torch import nn

from model.detector import StressDetector
from model.intervention import InterventionConfig, InterventionDecision, InterventionPolicy
from model.session import HostStressSession, WindowOutcome
from model.telemetry import ConfidenceReadout


class _ConstScore(nn.Module):
    """Stub model returning a fixed score, so detector state is deterministic."""

    def __init__(self, score: float) -> None:
        super().__init__()
        self.score = score

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.full((x.shape[0], 1), self.score)


_WAVE = torch.zeros(48000)


def test_outcome_bundles_state_confidence_and_intervention():
    sess = HostStressSession(_ConstScore(0.9))
    out = sess.process_window(_WAVE, voiced=True)
    assert isinstance(out, WindowOutcome)
    assert out.state is sess.state
    assert isinstance(out.confidence, ConfidenceReadout)
    assert isinstance(out.intervention, InterventionDecision)


def test_unvoiced_window_has_no_confidence_and_never_fires():
    sess = HostStressSession(_ConstScore(0.9))
    out = sess.process_window(_WAVE, voiced=False)
    assert out.confidence is None
    assert out.intervention.fire is False
    assert out.state.stressed is False


def test_sustained_stress_eventually_fires_the_cue():
    sess = HostStressSession(
        _ConstScore(0.95),
        intervention=InterventionPolicy(InterventionConfig(sustain_windows=3, cooldown_windows=10)),
    )
    fired = any(
        sess.process_window(_WAVE, voiced=True).intervention.fire for _ in range(12)
    )
    assert fired is True


def test_calm_stream_never_fires():
    sess = HostStressSession(_ConstScore(0.05))
    fired = any(
        sess.process_window(_WAVE, voiced=True).intervention.fire for _ in range(12)
    )
    assert fired is False


def test_dismiss_reaches_policy():
    pol = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=8))
    sess = HostStressSession(_ConstScore(0.95), intervention=pol)
    # drive to a fire, then dismiss
    for _ in range(6):
        sess.process_window(_WAVE, voiced=True)
    sess.dismiss()
    assert pol.in_cooldown is True


def test_reset_clears_pipeline_and_policy():
    sess = HostStressSession(_ConstScore(0.95))
    for _ in range(6):
        sess.process_window(_WAVE, voiced=True)
    sess.reset()
    assert sess.state.raw_score is None
    assert sess.policy.in_cooldown is False


def test_respects_injected_detector_and_temperature():
    det = StressDetector()
    sess = HostStressSession(_ConstScore(0.9), detector=det, temperature=1.0)
    out = sess.process_window(_WAVE, voiced=True)
    # temperature=1.0 is identity: calibrated probability equals the raw score.
    assert abs(out.confidence.calibrated_probability - out.state.raw_score) < 1e-6
