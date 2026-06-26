package ai.slashh.audio

/**
 * The on-device decision loop: one audio window in, one [StressState] out.
 *
 *   window → VAD gate → log-mel → classifier → EMA smoothing → hysteresis
 *
 * - **VAD gate**: unvoiced windows don't run the model (energy) and don't move
 *   the meter; we require ≥1 voiced window before reporting a level (plan §12).
 * - **EMA**: `EMA_ALPHA`-smoothed score so the meter is steady, not jittery.
 * - **Hysteresis**: enter "stressed" at `STRESS_THRESHOLD`, leave only below
 *   `RELEASE_THRESHOLD`, so the state doesn't flicker at the boundary (M8).
 *
 * Pure and synchronous: feed it [AudioCapture] windows; it owns no threads.
 */
class StressPipeline(
    private val classifier: StressClassifier,
    private val vad: Vad = Vad(),
    private val logMel: LogMel = LogMel(),
) {
    private var ema: Float = Float.NaN     // NaN until the first voiced window
    private var stressed = false
    private var voicedSeen = false

    data class StressState(
        val voiced: Boolean,
        /** raw model score for this window, or null if gated/silent */
        val rawScore: Float?,
        /** EMA-smoothed score, or null before any voiced window */
        val level: Float?,
        /** hysteresis latch — true while in the stressed band */
        val stressed: Boolean,
    )

    /** Process one [AudioConfig.WINDOW_SAMPLES] window. */
    fun onWindow(pcm: FloatArray): StressState {
        if (!vad.isVoiced(pcm)) {
            return StressState(
                voiced = false,
                rawScore = null,
                level = if (voicedSeen) ema else null,
                stressed = stressed,
            )
        }
        voicedSeen = true

        val features = logMel.extractFlat(pcm)
        val raw = classifier.score(features).coerceIn(0f, 1f)

        ema = if (ema.isNaN()) raw else AudioConfig.EMA_ALPHA * raw + (1 - AudioConfig.EMA_ALPHA) * ema

        // hysteresis: separate enter/exit thresholds
        stressed = when {
            ema >= AudioConfig.STRESS_THRESHOLD -> true
            ema < AudioConfig.RELEASE_THRESHOLD -> false
            else -> stressed
        }

        return StressState(voiced = true, rawScore = raw, level = ema, stressed = stressed)
    }

    /** Reset smoothing/latch state (e.g. after the screen is backgrounded). */
    fun reset() {
        ema = Float.NaN
        stressed = false
        voicedSeen = false
    }
}
