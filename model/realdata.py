"""Real labelled-speech datasets → binary calm/stressed (arousal) features.

The synthetic generator in ``model/data.py`` validates the pipeline; this module
loads *real* emotional-speech corpora and maps their categorical emotion labels
onto the single **arousal** axis (Russell's circumplex): high-activation states
(anger, fear, surprise, high-energy happy, disgust) → ``stressed`` (1.0);
low-activation states (neutral, calm, sad, boredom) → ``calm`` (0.0).

Two no-login corpora are supported out of the box:

* **RAVDESS speech** (Zenodo 1188976) — 24 actors, filename field 3 = emotion,
  field 7 = actor id. The only common corpus with an explicit ``calm`` label.
* **EMO-DB / Berlin** (emodb.bilderbar.info) — 10 actors, native 16 kHz mono;
  filename chars 1-2 = speaker, char 6 = emotion code.

Each clip is loaded, downmixed to mono, resampled to ``SAMPLE_RATE``, and
centre-cropped / padded to one ``WINDOW_SAMPLES`` window — the same fixed window
the exported ``.pte`` consumes — then run through the golden log-mel front-end.
Every example carries a ``speaker`` id so callers can build speaker-independent
splits (no actor in both train and eval), which is essential: emotion models
otherwise learn *who* is talking, not *how aroused* they are.

Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .audio_config import SAMPLE_RATE, WINDOW_SAMPLES
from .features import extract_batch

# --- Emotion → arousal maps (1.0 = stressed/high-arousal, 0.0 = calm/low) ----

# RAVDESS filename field 3 (1-indexed emotion code).
RAVDESS_AROUSAL: dict[str, float] = {
    "01": 0.0,  # neutral  -> calm
    "02": 0.0,  # calm     -> calm
    "03": 1.0,  # happy    -> stressed (high arousal)
    "04": 0.0,  # sad      -> calm
    "05": 1.0,  # angry    -> stressed
    "06": 1.0,  # fearful  -> stressed
    "07": 1.0,  # disgust  -> stressed
    "08": 1.0,  # surprised-> stressed
}

# EMO-DB single-char emotion code (filename position 6).
EMODB_AROUSAL: dict[str, float] = {
    "W": 1.0,  # Ärger/anger      -> stressed
    "A": 1.0,  # Angst/fear       -> stressed
    "F": 1.0,  # Freude/happiness -> stressed
    "E": 1.0,  # Ekel/disgust     -> stressed
    "N": 0.0,  # neutral          -> calm
    "L": 0.0,  # Langeweile/bored -> calm
    "T": 0.0,  # Trauer/sadness   -> calm
}


@dataclass(frozen=True)
class Clip:
    """One labelled audio clip before feature extraction."""

    path: Path
    label: float          # 1.0 stressed / 0.0 calm
    speaker: str          # corpus-qualified speaker id, e.g. "ravdess:12"
    emotion: str          # raw emotion code, for auditing


def _read_pcm_wav(path: str | Path) -> tuple[torch.Tensor, int]:
    """Read a 16/8/32-bit PCM wav → (float mono [samples], sample_rate).

    Uses the stdlib ``wave`` module so no audio codec / torchaudio backend is
    required (RAVDESS and EMO-DB are plain PCM wavs).
    """
    import wave as wavmod

    with wavmod.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    import numpy as np

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported PCM width {width*8}-bit in {path}")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:                                        # 8-bit is unsigned
        data = (data - 128.0) / 128.0
    else:
        data = data / float(1 << (8 * width - 1))         # int -> [-1, 1)
    if n_ch > 1:                                          # interleaved -> mono
        data = data.reshape(-1, n_ch).mean(axis=1)
    return torch.from_numpy(data.copy()), sr


def load_wav_window(path: str | Path, _cache: dict | None = None) -> torch.Tensor:
    """Load a wav → mono, 16 kHz, centre-cropped/padded to one window."""
    import torchaudio

    wave, sr = _read_pcm_wav(path)                       # mono float [samples]
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    n = wave.numel()
    if n >= WINDOW_SAMPLES:                               # centre crop
        start = (n - WINDOW_SAMPLES) // 2
        wave = wave[start:start + WINDOW_SAMPLES]
    else:                                                 # pad tail with zeros
        wave = torch.nn.functional.pad(wave, (0, WINDOW_SAMPLES - n))
    peak = wave.abs().max().clamp(min=1e-6)
    return wave / peak                                    # peak-normalize


def scan_ravdess(root: str | Path) -> list[Clip]:
    """Collect RAVDESS speech clips: ``03-01-EE-II-SS-RR-AA.wav``."""
    root = Path(root)
    clips: list[Clip] = []
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        emo, actor = parts[2], parts[6]
        if emo not in RAVDESS_AROUSAL:
            continue
        clips.append(Clip(wav, RAVDESS_AROUSAL[emo], f"ravdess:{actor}", emo))
    return clips


def scan_emodb(root: str | Path) -> list[Clip]:
    """Collect EMO-DB clips: ``SSxxxEv.wav`` (SS speaker, char[5] emotion)."""
    root = Path(root)
    clips: list[Clip] = []
    for wav in sorted(root.rglob("*.wav")):
        name = wav.stem
        if len(name) < 7:
            continue
        speaker, emo = name[:2], name[5]
        if emo not in EMODB_AROUSAL:
            continue
        clips.append(Clip(wav, EMODB_AROUSAL[emo], f"emodb:{speaker}", emo))
    return clips


SCANNERS = {"ravdess": scan_ravdess, "emodb": scan_emodb}


def scan_corpora(specs: dict[str, str | Path]) -> list[Clip]:
    """Scan multiple corpora. ``specs`` maps corpus-name -> root dir."""
    clips: list[Clip] = []
    for name, root in specs.items():
        if name not in SCANNERS:
            raise ValueError(f"unknown corpus {name!r}; known: {sorted(SCANNERS)}")
        found = SCANNERS[name](root)
        if not found:
            raise FileNotFoundError(f"no usable clips for {name!r} under {root}")
        clips.extend(found)
    return clips


@dataclass(frozen=True)
class RealDataset:
    x: torch.Tensor          # [N, 1, N_MELS, N_FRAMES]
    y: torch.Tensor          # [N, 1] in {0., 1.}
    speakers: list[str]      # length N
    emotions: list[str]      # length N (raw codes, for auditing)

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def stressed_fraction(self) -> float:
        return float(self.y.mean())


def build_real_dataset(
    specs: dict[str, str | Path],
    *,
    batch: int = 64,
    limit: int | None = None,
) -> RealDataset:
    """Load corpora → extracted log-mel features + labels + speaker ids.

    Clips are feature-extracted in batches of ``batch`` windows through the
    vectorized front-end. ``limit`` caps the clip count (smoke tests).
    """
    clips = scan_corpora(specs)
    if limit is not None:
        clips = clips[:limit]
    feats: list[torch.Tensor] = []
    for i in range(0, len(clips), batch):
        chunk = clips[i:i + batch]
        waves = torch.stack([load_wav_window(c.path) for c in chunk])
        feats.append(extract_batch(waves))
    x = torch.cat(feats, dim=0)
    y = torch.tensor([c.label for c in clips], dtype=torch.float32).unsqueeze(1)
    return RealDataset(
        x=x, y=y,
        speakers=[c.speaker for c in clips],
        emotions=[c.emotion for c in clips],
    )
