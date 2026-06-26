"""End-to-end host session — the full on-device loop, host-side.

Three host components otherwise live in isolation:

    HostStressPipeline   waveform -> log-mel -> StressNet -> StressDetector -> StressState
    telemetry.readout    StressState -> calibrated, NON-gating ConfidenceReadout
    InterventionPolicy   StressState stream -> when to surface a calming cue

``HostStressSession`` composes them so one call per window yields the whole
outcome — state, confidence read-out, and intervention decision — mirroring what
the device loop produces each hop. The detector gate is still the only thing that
decides ``stressed``; confidence is display-only and the intervention policy
consumes the gate's decision without altering it.

    from model.session import HostStressSession
    sess = HostStressSession(model)
    out = sess.process_window(waveform, voiced=True)
    if out.intervention.fire:
        show_calming_cue()        # and call sess.dismiss() when the user dismisses

Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .confidence import DEFAULT_TEMPERATURE
from .detector import StressDetector, StressState
from .intervention import InterventionDecision, InterventionPolicy
from .pipeline import HostStressPipeline
from .telemetry import ConfidenceReadout, readout


@dataclass(frozen=True)
class WindowOutcome:
    """Everything one window produces: the decided state, its (display-only)
    confidence read-out, and the calming-cue decision."""

    state: StressState
    confidence: ConfidenceReadout | None
    intervention: InterventionDecision


class HostStressSession:
    """Compose the pipeline, confidence telemetry, and intervention policy."""

    def __init__(
        self,
        model: nn.Module,
        *,
        detector: StressDetector | None = None,
        intervention: InterventionPolicy | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.pipeline = HostStressPipeline(model, detector)
        self.policy = intervention or InterventionPolicy()
        self.temperature = temperature

    @torch.no_grad()
    def process_window(self, waveform: torch.Tensor, *, voiced: bool = True) -> WindowOutcome:
        """Run one window end to end and bundle the outcome."""
        state = self.pipeline.process_window(waveform, voiced=voiced)
        conf = readout(state, temperature=self.temperature)
        decision = self.policy.update(state)
        return WindowOutcome(state=state, confidence=conf, intervention=decision)

    @property
    def state(self) -> StressState:
        """The most recent decided state."""
        return self.pipeline.state

    def dismiss(self) -> None:
        """User dismissed the calming cue — propagate to the policy's cooldown."""
        self.policy.dismiss()

    def reset(self) -> None:
        """Clear pipeline and policy state (e.g. new session)."""
        self.pipeline.reset()
        self.policy.reset()
