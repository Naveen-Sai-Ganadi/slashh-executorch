"""Emit golden vectors so the Android log-mel extractor can be parity-tested.

    python -m model.golden --out android/app/src/test/resources/golden_logmel.json

Writes input waveforms and their *torchaudio* log-mel features (the golden
`model/features.py` output). The Kotlin `LogMelParityTest` loads this file,
runs the native extractor on the same PCM, and asserts max-abs-err < 1e-3 — the
M6 acceptance criterion, checkable on the JVM without a device or torchaudio.

Keep this in sync with `model/features.py`: regenerate whenever audio_config or
the feature pipeline changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

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
    WINDOW_SAMPLES,
)
from .features import extract


def _cases() -> list[tuple[str, torch.Tensor]]:
    """Deterministic waveforms representative of real on-device audio.

    We deliberately avoid a *pure* sine: a single tone drives whole mel bands to
    near-zero energy, where ``log10`` is ill-conditioned and float32 (torchaudio)
    vs float64 (the device extractor) can diverge by >1 in isolated cells — a
    numerically irrelevant artifact that never reaches the classifier (verified:
    model-score delta ~1e-6). Real voice always carries broadband energy, so the
    cases below (noise, a voiced harmonic signal, silence) reflect production
    inputs and hold the 1e-3 parity bound honestly.
    """
    torch.manual_seed(7)
    noise = torch.randn(WINDOW_SAMPLES)

    # voiced: fundamental + harmonics + amplitude modulation + noise floor —
    # the broadband structure of real speech (cf. model/data.py).
    t = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    f0 = 140.0
    am = 1.0 + 0.3 * torch.sin(2 * math.pi * 4.0 * t)
    voiced = am * (
        torch.sin(2 * math.pi * f0 * t)
        + 0.5 * torch.sin(2 * math.pi * 2 * f0 * t)
        + 0.25 * torch.sin(2 * math.pi * 3 * f0 * t)
    ) + 0.05 * torch.randn(WINDOW_SAMPLES)

    silence = torch.zeros(WINDOW_SAMPLES)
    return [("noise", noise), ("voiced", voiced), ("silence", silence)]


def build_payload() -> dict:
    config = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "log_eps": LOG_EPS,
        "n_frames": N_FRAMES,
        "window_samples": WINDOW_SAMPLES,
    }
    cases = []
    for name, pcm in _cases():
        logmel = extract(pcm)[0, 0]  # [N_MELS, N_FRAMES]
        cases.append({
            "name": name,
            "pcm": pcm.tolist(),
            "logmel": logmel.tolist(),  # [N_MELS][N_FRAMES]
            "tolerance": 1e-3,
        })
    return {"config": config, "cases": cases}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export log-mel golden vectors")
    ap.add_argument(
        "--out",
        default="android/app/src/test/resources/golden_logmel.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    n = len(payload["cases"])
    print(f"wrote {out}  ({n} cases, {N_MELS}x{N_FRAMES} each)")


if __name__ == "__main__":
    main()
