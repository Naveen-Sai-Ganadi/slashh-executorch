"""Datasets for the calm-vs-stressed classifier.

Two sources:

1. **Synthetic** (default, offline) — structured proxy signals so the *whole*
   train → export → run pipeline works today without any download. "Stressed"
   windows carry more high-band energy and stronger amplitude modulation
   (a crude analog of the tense, higher-pitched, less-steady voice of arousal);
   "calm" windows are lower, smoother. The CNN learns to separate them, which
   validates the architecture, the loss, and feature parity end to end.

2. **Folder** — real labelled speech (RAVDESS/CREMA-D/TESS/SAVEE, mapped to
   arousal per plan §13). Point ``--data-dir`` at:
       <root>/calm/*.wav
       <root>/stressed/*.wav
   Swap synthetic → folder once data is on disk; nothing else changes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .audio_config import SAMPLE_RATE, WINDOW_SAMPLES
from .features import extract, extract_batch


def _synth_waveform(stressed: bool, gen: torch.Generator) -> torch.Tensor:
    """One WINDOW_SAMPLES proxy waveform for the given class."""
    t = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    if stressed:
        f0 = 220.0 + 80.0 * torch.rand(1, generator=gen).item()     # higher pitch
        mod_hz = 6.0 + 4.0 * torch.rand(1, generator=gen).item()    # fast tremor
        mod = 1.0 + 0.5 * torch.sin(2 * torch.pi * mod_hz * t)      # strong AM
        harm = 0.4 * torch.sin(2 * torch.pi * 3 * f0 * t)          # high harmonics
        noise = 0.20
    else:
        f0 = 110.0 + 40.0 * torch.rand(1, generator=gen).item()     # lower pitch
        mod = 1.0 + 0.05 * torch.sin(2 * torch.pi * 2.0 * t)        # steady
        harm = 0.1 * torch.sin(2 * torch.pi * 2 * f0 * t)
        noise = 0.05
    wave = mod * torch.sin(2 * torch.pi * f0 * t) + harm
    wave = wave + noise * torch.randn(WINDOW_SAMPLES, generator=gen)
    return wave / wave.abs().max().clamp(min=1e-6)


def synthetic_dataset(
    n_per_class: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features [N,1,N_MELS,N_FRAMES], labels [N,1]) — balanced."""
    gen = torch.Generator().manual_seed(seed)
    waves, labels = [], []
    for stressed in (False, True):
        for _ in range(n_per_class):
            waves.append(_synth_waveform(stressed, gen))  # RNG order preserved
            labels.append(float(stressed))
    # One vectorized front-end call over the whole batch instead of N loop
    # calls to extract() — same features (allclose), much faster to build.
    x = extract_batch(torch.stack(waves))              # [N,1,N_MELS,N_FRAMES]
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


def folder_dataset(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load <root>/calm/*.wav and <root>/stressed/*.wav into features+labels."""
    import torchaudio

    root = Path(root)
    feats, labels = [], []
    for label, sub in ((0.0, "calm"), (1.0, "stressed")):
        for wav_path in sorted((root / sub).glob("*.wav")):
            wave, sr = torchaudio.load(wav_path)
            wave = wave.mean(0)                          # mono
            if sr != SAMPLE_RATE:
                wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
            # center-crop / pad to one window
            if wave.numel() >= WINDOW_SAMPLES:
                start = (wave.numel() - WINDOW_SAMPLES) // 2
                wave = wave[start:start + WINDOW_SAMPLES]
            else:
                wave = torch.nn.functional.pad(
                    wave, (0, WINDOW_SAMPLES - wave.numel())
                )
            feats.append(extract(wave))
            labels.append(label)
    if not feats:
        raise FileNotFoundError(
            f"no .wav under {root}/calm or {root}/stressed"
        )
    x = torch.cat(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y
