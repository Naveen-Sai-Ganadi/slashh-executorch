package ai.slashh.audio

/**
 * Single source of truth for audio constants — the Kotlin mirror of
 * `model/audio_config.py`. These MUST stay byte-for-byte equal to the Python
 * side; training-time and inference-time features only agree if every constant
 * here matches (plan §11). If you change one, change both and regenerate the
 * golden vectors (`python -m model.golden`).
 */
object AudioConfig {
    const val SAMPLE_RATE = 16_000
    const val WINDOW_SECONDS = 3.0
    const val HOP_SECONDS = 1.0

    const val N_FFT = 400
    const val WIN_LENGTH = 400
    const val HOP_LENGTH = 160
    const val N_MELS = 64
    const val F_MIN = 0.0
    const val F_MAX = 8_000.0
    const val LOG_EPS = 1e-6

    const val WINDOW_SAMPLES = 48_000          // WINDOW_SECONDS * SAMPLE_RATE
    const val HOP_SAMPLES = 16_000             // HOP_SECONDS * SAMPLE_RATE
    const val N_FRAMES = 301                   // 1 + WINDOW_SAMPLES / HOP_LENGTH

    // Stress meter: hysteresis so the level doesn't flicker at the boundary.
    const val STRESS_THRESHOLD = 0.6f
    const val RELEASE_THRESHOLD = 0.45f
    const val EMA_ALPHA = 0.4f
}
