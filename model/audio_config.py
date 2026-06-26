"""Single source of truth for audio + feature parameters.

These constants MUST match bit-for-bit between:
  - offline training/eval (this package, torchaudio),
  - the exported ``.pte`` input shape, and
  - the on-device Android feature extractor (Kotlin/C++).

If any of these change, training, the model input shape, and the device-side
DSP all change together. Treat this file as a frozen spec during the event.
(Plan §3, §11: log-mel param mismatch silently collapses accuracy.)
"""

from __future__ import annotations

# --- Raw audio ---
SAMPLE_RATE: int = 16_000           # mono PCM, 16 kHz
WINDOW_SECONDS: float = 3.0         # rolling classification window
HOP_SECONDS: float = 1.0            # stride between windows (live inference)

# --- Log-mel spectrogram (torchaudio MelSpectrogram, power=2, center=True) ---
N_FFT: int = 400                    # 25 ms @ 16 kHz
WIN_LENGTH: int = 400               # 25 ms
HOP_LENGTH: int = 160               # 10 ms
N_MELS: int = 64
F_MIN: float = 0.0
F_MAX: float = 8_000.0              # Nyquist for 16 kHz
LOG_EPS: float = 1e-6               # floor before log10

# --- Derived fixed model input shape -------------------------------------
# torchaudio MelSpectrogram with center=True yields 1 + floor(n_samples/hop)
# frames. For a 3.0 s window: 48000 samples -> 1 + 300 = 301 frames.
WINDOW_SAMPLES: int = int(round(WINDOW_SECONDS * SAMPLE_RATE))   # 48000
N_FRAMES: int = 1 + WINDOW_SAMPLES // HOP_LENGTH                 # 301

# The exported .pte consumes exactly this shape: [batch, channel, mels, frames].
MODEL_INPUT_SHAPE: tuple[int, int, int, int] = (1, 1, N_MELS, N_FRAMES)

# --- Inference / UX defaults (overridable on device) ---
STRESS_THRESHOLD: float = 0.6       # enter "stressed"
RELEASE_THRESHOLD: float = 0.45     # exit hysteresis (< enter, avoids flicker)
EMA_ALPHA: float = 0.4              # smoothing weight on the newest window
