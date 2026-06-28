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

    // Hysteresis in MAPPED [0,1] stress space: enter high, leave low. Tuned conservative
    // (0.62 / 0.45) so only sustained, clearly-elevated fused stress latches "stressed" —
    // fewer false positives on ordinary speech. Pairs with the fusion's a+t>=1.2 boundary.
    var enterThreshold: Float = 0.62f
    var releaseThreshold: Float = 0.45f

    // Optional late fusion with a text-stress signal (Whisper transcript → text model).
    // When BOTH are set and a fresh text score is available for this window, the audio
    // stress is fused with the text score and that fused probability drives the meter
    // directly (it is already a calibrated [0,1] probability). When either is null we
    // behave exactly as before — audio-only. This is the "more confident detection from
    // both features" path; absent a transcript it degrades honestly to audio alone.
    var textScoreProvider: (() -> Float?)? = null
    var fuse: ((audioStress: Float, textScore: Float) -> Float)? = null

    // Optional neural audio model (WavLM) scored asynchronously on the NPU — too slow
    // (~8 s/forward) for the synchronous per-window path, so it's polled here for its freshest
    // value. When present it becomes the audio leg of the fusion (the genuine on-NPU audio
    // score); energy still drives the meter moment-to-moment so it stays responsive.
    var audioModelProvider: (() -> Float?)? = null

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
        /** audio-only stress for this window in [0,1] (pre-fusion), or null if gated */
        val audioStress: Float? = null,
        /** latest text-stress score fused in for this window, or null if none/stale */
        val textScore: Float? = null,
        /** fused audio+text probability for this window, or null when audio-only */
        val fused: Float? = null,
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

        // Fast, responsive audio signal: energy (RMS) mapped between the calm/stress anchors
        // (or the synchronous model score if calibrated to it). This drives the meter every hop.
        val signal = if (useModelSignal) raw else vad.lastRms.toFloat()
        val energyStress = toStress(signal)

        // WavLM neural audio score (on the NPU, scored async + EMA-smoothed in the coordinator).
        // It is the fusion's audio leg — the energy signal is only the fallback for the brief
        // windows before the first WavLM score / if the helper stalls.
        val neuralAudio = audioModelProvider?.invoke()?.coerceIn(0f, 1f)
        val audioStress = neuralAudio ?: energyStress

        // Late fusion: combine the audio stress with the latest text-stress score (if a fresh
        // transcript produced one). No text → audio-only.
        val text = textScoreProvider?.invoke()
        val fuser = fuse
        val fused = if (text != null && fuser != null) fuser(audioStress, text).coerceIn(0f, 1f) else null

        // The meter/latch follows the fused score when text is present, else the fast energy
        // signal — so it stays responsive between the slow WavLM/Whisper NPU updates.
        val combined = fused ?: energyStress

        ema = if (ema.isNaN()) combined else AudioConfig.EMA_ALPHA * combined + (1 - AudioConfig.EMA_ALPHA) * ema

        // hysteresis in mapped stress space: separate enter/exit thresholds
        stressed = when {
            ema >= enterThreshold -> true
            ema < releaseThreshold -> false
            else -> stressed
        }

        // rawScore surfaces the genuine on-NPU WavLM score when available (the audio-model
        // evidence / badge), else the synchronous model output.
        return StressState(
            voiced = true, rawScore = neuralAudio ?: raw, level = ema, stressed = stressed,
            audioStress = audioStress, textScore = text, fused = fused,
        )
    }

    /** Reset smoothing/latch state (e.g. after the screen is backgrounded). */
    fun reset() {
        ema = Float.NaN
        stressed = false
        voicedSeen = false
    }
}
