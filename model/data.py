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
from .features import extract


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
    feats, labels = [], []
    for stressed in (False, True):
        for _ in range(n_per_class):
            wave = _synth_waveform(stressed, gen)
            feats.append(extract(wave))                 # [1,1,N_MELS,N_FRAMES]
            labels.append(float(stressed))
    x = torch.cat(feats, dim=0)                         # [N,1,N_MELS,N_FRAMES]
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


def _load_wave(wav_path: str | Path) -> torch.Tensor:
    """Load a wav as a mono, 16 kHz, single-window float waveform.

    Uses soundfile, not torchaudio.load: torchaudio 2.11 routes load() through
    torchcodec, which isn't installed — soundfile is the working path here.
    """
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32")     # [T] or [T, ch]
    wave = torch.from_numpy(data)
    if wave.dim() == 2:
        wave = wave.mean(1)                                # mono
    import torchaudio
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    if wave.numel() >= WINDOW_SAMPLES:
        start = (wave.numel() - WINDOW_SAMPLES) // 2
        wave = wave[start:start + WINDOW_SAMPLES]
    else:
        wave = torch.nn.functional.pad(wave, (0, WINDOW_SAMPLES - wave.numel()))
    return wave


def folder_dataset(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load <root>/calm/*.wav and <root>/stressed/*.wav into features+labels."""
    root = Path(root)
    feats, labels = [], []
    for label, sub in ((0.0, "calm"), (1.0, "stressed")):
        for wav_path in sorted((root / sub).glob("*.wav")):
            feats.append(extract(_load_wave(wav_path)))
            labels.append(label)
    if not feats:
        raise FileNotFoundError(f"no .wav under {root}/calm or {root}/stressed")
    x = torch.cat(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y


# RAVDESS emotion code (filename field[2]) -> arousal class, per plan §13:
# "high-arousal (angry/fearful) -> stressed, calm/neutral -> calm".
RAVDESS_STRESSED = {"05", "06"}          # angry, fearful
RAVDESS_CALM = {"01", "02"}             # neutral, calm
# happy/sad/disgust/surprised (03/04/07/08) intentionally excluded — they are
# not cleanly "stress vs calm" and would blur a stress classifier.


def ravdess_dataset(
    root: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load RAVDESS Actor_*/ *.wav directly into (features, labels, actor_ids).

    Returns actor ids so callers can build a SPEAKER-INDEPENDENT split (no actor
    appears in both train and val) — the honest way to report accuracy.
    RAVDESS filename: ``03-01-06-01-02-01-12.wav`` -> field[2]=emotion, field[6]=actor.
    """
    root = Path(root)
    wavs = sorted(root.rglob("*.wav"))
    feats, labels, actors = [], [], []
    for wav_path in wavs:
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            continue
        emotion, actor = parts[2], int(parts[6])
        if emotion in RAVDESS_STRESSED:
            label = 1.0
        elif emotion in RAVDESS_CALM:
            label = 0.0
        else:
            continue
        feats.append(extract(_load_wave(wav_path)))
        labels.append(label)
        actors.append(actor)
    if not feats:
        raise FileNotFoundError(f"no RAVDESS Actor_*/ *.wav under {root}")
    x = torch.cat(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    a = torch.tensor(actors, dtype=torch.long)
    return x, y, a
