"""Feature-extraction contract tests (plan §11).

The on-device Android extractor must reproduce these shapes and parameters, and
training/inference must see identical features. These tests pin the contract:
fixed output shape, determinism, and graceful handling of off-length input.
"""

from __future__ import annotations

import torch

from model.audio_config import (
    MODEL_INPUT_SHAPE,
    N_FRAMES,
    N_MELS,
    WINDOW_SAMPLES,
)
from model.features import extract


def test_output_shape_is_fixed_contract():
    pcm = torch.zeros(WINDOW_SAMPLES)
    out = extract(pcm)
    assert tuple(out.shape) == MODEL_INPUT_SHAPE == (1, 1, N_MELS, N_FRAMES)


def test_extract_is_deterministic():
    torch.manual_seed(0)
    pcm = torch.randn(WINDOW_SAMPLES)
    a = extract(pcm)
    b = extract(pcm.clone())
    assert torch.equal(a, b)


def test_short_and_long_input_yield_fixed_frames():
    short = extract(torch.randn(WINDOW_SAMPLES // 2))
    long = extract(torch.randn(WINDOW_SAMPLES * 2))
    assert short.shape[-1] == N_FRAMES
    assert long.shape[-1] == N_FRAMES


def test_rejects_non_1d_input():
    try:
        extract(torch.zeros(2, WINDOW_SAMPLES))
    except ValueError:
        return
    raise AssertionError("expected ValueError on 2-D input")


def test_features_are_finite():
    pcm = torch.randn(WINDOW_SAMPLES)
    out = extract(pcm)
    assert torch.isfinite(out).all(), "log-mel produced non-finite values"
