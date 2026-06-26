"""Batched log-mel front-end (``extract_batch``) contract + parity tests.

The single-window :func:`model.features.extract` is the golden reference, but
every dataset / robustness / A-B builder computes features in a Python ``for``
loop — one MelSpectrogram call per window, hundreds per run. ``extract_batch``
vectorizes that front-end across a batch in a single call. It must produce the
SAME features as stacking per-window ``extract`` (allclose — the batched mel
matmul reorders reductions, so equality holds only to ~1e-6), the same fixed
output contract, and reject non-2-D input. Host-only; the device extractor is
unaffected (it still streams one window at a time).
"""

from __future__ import annotations

import torch

from model.audio_config import N_FRAMES, N_MELS, WINDOW_SAMPLES
from model.features import extract, extract_batch


def test_batch_matches_per_sample_extract():
    gen = torch.Generator().manual_seed(0)
    waves = [torch.randn(WINDOW_SAMPLES, generator=gen) for _ in range(5)]
    batched = extract_batch(torch.stack(waves))
    per_sample = torch.cat([extract(w) for w in waves], dim=0)
    assert batched.shape == per_sample.shape
    assert torch.allclose(batched, per_sample, atol=1e-5, rtol=1e-4)


def test_batch_shape_contract():
    out = extract_batch(torch.zeros(7, WINDOW_SAMPLES))
    assert tuple(out.shape) == (7, 1, N_MELS, N_FRAMES)


def test_single_row_batch_matches_extract():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(WINDOW_SAMPLES, generator=gen)
    one = extract_batch(w.unsqueeze(0))
    assert torch.allclose(one[0], extract(w), atol=1e-5, rtol=1e-4)


def test_rejects_non_2d_input():
    for bad in (torch.zeros(WINDOW_SAMPLES), torch.zeros(2, 1, WINDOW_SAMPLES)):
        try:
            extract_batch(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on shape {tuple(bad.shape)}")


def test_batch_output_is_contiguous():
    # The ExecuTorch runtime rejects non-contiguous forward inputs, and these
    # features feed straight into exported .pte programs — a non-contiguous
    # reshape view here breaks every export/quantization path downstream.
    out = extract_batch(torch.zeros(5, WINDOW_SAMPLES))
    assert out.is_contiguous()


def test_batch_is_deterministic():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(4, WINDOW_SAMPLES, generator=gen)
    assert torch.equal(extract_batch(x), extract_batch(x.clone()))


def test_batch_features_are_finite():
    gen = torch.Generator().manual_seed(3)
    out = extract_batch(torch.randn(3, WINDOW_SAMPLES, generator=gen))
    assert torch.isfinite(out).all(), "batched log-mel produced non-finite values"
