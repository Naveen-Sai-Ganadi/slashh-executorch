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
    # Per-window standardization: makes features amplitude/mic-invariant so the
    # model keys on spectral SHAPE (tone), not absolute level — essential for
    # real on-device mic input. MUST match LogMel.kt bit-for-bit (population std).
    logmel = (logmel - logmel.mean()) / (logmel.std(unbiased=False) + 1e-5)
    return logmel.reshape(1, 1, N_MELS, N_FRAMES)
