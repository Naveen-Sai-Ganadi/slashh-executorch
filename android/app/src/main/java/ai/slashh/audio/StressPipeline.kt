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
    /** When set, voiced windows are scored directly from raw pcm (e.g. the WavLM
     *  teacher on the NPU, which normalizes + extracts features in-graph) and the
     *  host-side log-mel + [classifier] path is skipped. Null = the StressNet path. */
    private val rawScorer: RawWaveScorer? = null,
) {
    private var ema: Float = Float.NaN     // NaN until the first voiced window
    private var stressed = false
    private var voicedSeen = false

    // The stress level maps an input signal to [0,1] between two calibrated
    // anchors (user's calm -> 0, stressed -> 1). The signal is EITHER the model's
    // output (genuine ML stress, when the retrained model separates this voice)
    // OR vocal intensity (RMS energy, the robust fallback). "Calibrate to my
    // voice" measures both and picks whichever separates the demoer — set here.
    // Defaults: energy with sane anchors, so it works before calibration.
    var useModelSignal: Boolean = false
    var calmAnchor: Float = 0.03f
    var stressAnchor: Float = 0.14f

    // Hysteresis in MAPPED [0,1] stress space: enter high, leave low.
    var enterThreshold: Float = 0.55f
    var releaseThreshold: Float = 0.40f

    private fun toStress(signal: Float): Float {
        val span = stressAnchor - calmAnchor
        if (kotlin.math.abs(span) < 1e-4f) return 0f
        return ((signal - calmAnchor) / span).coerceIn(0f, 1f)
    }

    // VAD telemetry from the most recent window (on-device tuning/debug)
    val vadRms: Double get() = vad.lastRms
    val vadZcr: Double get() = vad.lastZcr
    val vadFloor: Double get() = vad.noiseFloor()

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

        val raw = (rawScorer?.scoreWindow(pcm)
            ?: classifier.score(logMel.extractFlat(pcm))).coerceIn(0f, 1f)   // model runs on-device
        val signal = if (useModelSignal) raw else vad.lastRms.toFloat()
        val stress = toStress(signal)

        ema = if (ema.isNaN()) stress else AudioConfig.EMA_ALPHA * stress + (1 - AudioConfig.EMA_ALPHA) * ema

        // hysteresis in mapped stress space: separate enter/exit thresholds
        stressed = when {
            ema >= enterThreshold -> true
            ema < releaseThreshold -> false
            else -> stressed
        }

        // rawScore carries the un-mapped model output (calibration needs it)
        return StressState(voiced = true, rawScore = raw, level = ema, stressed = stressed)
    }

    /** Reset smoothing/latch state (e.g. after the screen is backgrounded). */
    fun reset() {
        ema = Float.NaN
        stressed = false
        voicedSeen = false
    }
}
