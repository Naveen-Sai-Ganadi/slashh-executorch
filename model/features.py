"""Log-mel feature extraction (offline reference).

This is the *golden* implementation. The on-device Android extractor must match
these exact parameters (audio_config) so that training-time and inference-time
features agree — otherwise accuracy collapses (plan §11).

The model graph does NOT include this; features are computed outside the ``.pte``
and the resulting [1, 1, N_MELS, N_FRAMES] tensor is fed to the runtime.
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T

from .audio_config import (
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    LOG_EPS,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
)


class LogMelExtractor(torch.nn.Module):
    """waveform [.. , samples] -> log-mel [.. , N_MELS, N_FRAMES]."""

    def __init__(self) -> None:
        super().__init__()
        self.melspec = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            power=2.0,
            center=True,
            norm=None,
            mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.melspec(waveform)                 # [.., N_MELS, frames]
        logmel = torch.log10(mel + LOG_EPS)
        return logmel


_extractor: LogMelExtractor | None = None


def _shared_extractor() -> LogMelExtractor:
    global _extractor
    if _extractor is None:
        _extractor = LogMelExtractor().eval()
    return _extractor


def extract(pcm: torch.Tensor) -> torch.Tensor:
    """Single-window PCM -> model input tensor [1, 1, N_MELS, N_FRAMES].

    ``pcm`` is a 1-D float waveform of WINDOW_SAMPLES; shorter/longer inputs are
    padded/truncated to the fixed window so the output frame count is N_FRAMES.
    """
    if pcm.dim() != 1:
        raise ValueError(f"expected 1-D PCM, got shape {tuple(pcm.shape)}")
    with torch.no_grad():
        logmel = _shared_extractor()(pcm)            # [N_MELS, frames]
    # Guard the fixed frame count regardless of small input-length drift.
    frames = logmel.shape[-1]
    if frames < N_FRAMES:
        logmel = torch.nn.functional.pad(logmel, (0, N_FRAMES - frames))
    elif frames > N_FRAMES:
        logmel = logmel[..., :N_FRAMES]
    return logmel.reshape(1, 1, N_MELS, N_FRAMES)


def extract_batch(pcm: torch.Tensor) -> torch.Tensor:
    """Batched PCM -> model input ``[B, 1, N_MELS, N_FRAMES]``.

    ``pcm`` is ``[B, WINDOW_SAMPLES]`` — every row a fixed-length window. The
    log-mel front-end is vectorized across the batch in a *single*
    ``MelSpectrogram`` call instead of ``B`` Python-loop calls to
    :func:`extract`, which is the per-sample cost the dataset/robustness/A-B
    builders pay hundreds of times per run. The result matches stacking
    per-window :func:`extract` to ~1e-6 (the batched mel matmul reorders
    reductions, so it is allclose rather than bit-equal).

    This is a host-side throughput convenience; the on-device extractor is
    unaffected — it still streams one window at a time.
    """
    if pcm.dim() != 2:
        raise ValueError(f"expected 2-D [B, samples] PCM, got shape {tuple(pcm.shape)}")
    with torch.no_grad():
        logmel = _shared_extractor()(pcm)            # [B, N_MELS, frames]
    frames = logmel.shape[-1]
    if frames < N_FRAMES:
        logmel = torch.nn.functional.pad(logmel, (0, N_FRAMES - frames))
    elif frames > N_FRAMES:
        logmel = logmel[..., :N_FRAMES]
    # Match extract()'s effective contiguity (per-sample + torch.cat is always
    # contiguous): the ExecuTorch runtime rejects non-contiguous forward inputs,
    # and these features feed straight into exported .pte programs.
    return logmel.reshape(pcm.shape[0], 1, N_MELS, N_FRAMES).contiguous()
