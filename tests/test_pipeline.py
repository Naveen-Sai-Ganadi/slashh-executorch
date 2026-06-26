"""End-to-end host pipeline tests (model/pipeline.py).

This is the host mirror of the Android ``StressPipeline.kt``: it composes the
log-mel feature extractor, the trained StressNet, and the ``StressDetector``
(EMA + hysteresis) into one waveform -> state chain. The unit tests cover each
piece; this proves they work *together* — a regression in any link fails here.

The behavioural margins are large (calm scores ~0.14 vs stressed ~0.92,
verified by spike before these assertions were written), so this is not flaky.
Host-only; no device, no AI Hub, no token.
"""

from __future__ import annotations

import torch

from model.audio_config import STRESS_THRESHOLD
from model.data import _synth_waveform
from model.detector import StressState
from model.pipeline import HostStressPipeline
from model.production import train_production


def _model():
    torch.manual_seed(0)
    model, _ = train_production(epochs=12, n_per_class=96, seed=0)
    return model


def _trace(pipe, labels, *, voiced=True, seed=123):
    gen = torch.Generator().manual_seed(seed)
    states = []
    for stressed in labels:
        wave = _synth_waveform(stressed, gen)
        states.append(pipe.process_window(wave, voiced=voiced))
    return states


def test_calm_then_stressed_flips_the_latch() -> None:
    pipe = HostStressPipeline(_model())
    states = _trace(pipe, [False] * 6 + [True] * 6)

    calm, stressed = states[:6], states[6:]
    # never falsely latches during the calm segment
    assert not any(s.stressed for s in calm)
    # the calm meter sits well below the stress threshold
    assert all(s.level is not None and s.level < STRESS_THRESHOLD for s in calm)
    # the stressed segment latches and stays latched at the end
    assert stressed[-1].stressed
    assert stressed[-1].level > STRESS_THRESHOLD
    # raw scores separate cleanly (sanity on the model link)
    assert max(s.raw_score for s in calm) < min(s.raw_score for s in stressed)


def test_unvoiced_window_holds_state() -> None:
    pipe = HostStressPipeline(_model())
    _trace(pipe, [True] * 6)            # drive into the stressed latch
    before = pipe.state
    assert before.stressed

    # a gated (unvoiced) window must not run the model or change the latch
    held = pipe.process_window(_synth_waveform(False, torch.Generator().manual_seed(7)),
                               voiced=False)
    assert held.voiced is False
    assert held.raw_score is None
    assert held.stressed == before.stressed
    assert held.level == before.level


def test_reset_clears_state() -> None:
    pipe = HostStressPipeline(_model())
    _trace(pipe, [True] * 6)
    assert pipe.state.stressed
    pipe.reset()
    assert isinstance(pipe.state, StressState)
    assert pipe.state.stressed is False
    assert pipe.state.level is None
