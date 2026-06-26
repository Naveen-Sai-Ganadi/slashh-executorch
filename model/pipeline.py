"""Host stress pipeline — the waveform -> state chain, end to end.

This is the host-side mirror of the Android ``StressPipeline.kt``: it composes
the three host components that are otherwise only unit-tested in isolation —

    waveform -> features.extract (log-mel) -> StressNet (score) -> StressDetector
                                                          (EMA + hysteresis) -> state

into one object, so the *whole* chain is exercised and regressions in any link
surface immediately. The VAD lives on-device (Kotlin ``Vad.kt``); here the
``voiced`` gate is supplied per window, matching ``StressDetector.update``.

    from model.pipeline import HostStressPipeline
    pipe = HostStressPipeline(model)
    state = pipe.process_window(waveform, voiced=True)   # -> StressState

Host-only; no device, no AI Hub, no live token.
"""

from __future__ import annotations

import torch
from torch import nn

from .detector import StressDetector, StressState
from .features import extract

# Neutral state before any window is seen (mirrors a freshly-reset detector).
_NEUTRAL = StressState(voiced=False, raw_score=None, level=None, stressed=False)


class HostStressPipeline:
    """Compose feature extraction, the model, and the detector into one chain."""

    def __init__(self, model: nn.Module, detector: StressDetector | None = None) -> None:
        self.model = model.eval()
        self.detector = detector or StressDetector()
        self._last: StressState = _NEUTRAL

    @torch.no_grad()
    def process_window(self, waveform: torch.Tensor, *, voiced: bool = True) -> StressState:
        """Run one window: extract -> score -> detector. Gated windows skip the model."""
        if not voiced:
            self._last = self.detector.update(None, voiced=False)
            return self._last
        score = float(self.model(extract(waveform)).item())
        self._last = self.detector.update(score, voiced=True)
        return self._last

    def run(self, waveforms, voiced=None) -> list[StressState]:
        """Process a sequence of windows; ``voiced`` is a per-window bool list or None."""
        out = []
        for i, wave in enumerate(waveforms):
            v = True if voiced is None else bool(voiced[i])
            out.append(self.process_window(wave, voiced=v))
        return out

    @property
    def state(self) -> StressState:
        """The most recent state (neutral before any window / after reset)."""
        return self._last

    def reset(self) -> None:
        self.detector.reset()
        self._last = _NEUTRAL
