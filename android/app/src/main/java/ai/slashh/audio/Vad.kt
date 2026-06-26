package ai.slashh.audio

import kotlin.math.sqrt

/**
 * Lightweight voice-activity detector. Gating inference on voiced windows is the
 * energy-efficiency lever (plan §6, §8): silence never reaches the NPU, and we
 * never classify a quiet room as "stressed" (plan §12 — require ≥1 voiced window
 * before showing a level).
 *
 * Two cheap, robust cues:
 *  - **short-term energy** (RMS) above an adaptive noise floor, and
 *  - **zero-crossing rate** within a speech-plausible band (rejects pure hiss).
 *
 * The noise floor adapts downward quickly and upward slowly, so it tracks a
 * quiet room without being dragged up by speech.
 */
class Vad(
    private val energyMarginDb: Double = 6.0,   // how far above the floor counts as voice
    private val zcrMin: Double = 0.02,
    private val zcrMax: Double = 0.35,
) {
    private var noiseFloorRms = 1e-4            // adaptive; seeded low
    private var seeded = false

    /** Returns true if [window] (mono float PCM) looks like speech. */
    fun isVoiced(window: FloatArray): Boolean {
        if (window.isEmpty()) return false

        var sumSq = 0.0
        var crossings = 0
        var prev = window[0]
        for (i in window.indices) {
            val x = window[i]
            sumSq += x.toDouble() * x
            if (i > 0 && ((x >= 0f) != (prev >= 0f))) crossings++
            prev = x
        }
        val rms = sqrt(sumSq / window.size)
        val zcr = crossings.toDouble() / window.size

        if (!seeded) { noiseFloorRms = rms.coerceAtLeast(1e-6); seeded = true }

        val threshold = noiseFloorRms * Math.pow(10.0, energyMarginDb / 20.0)
        val voiced = rms > threshold && zcr in zcrMin..zcrMax

        // adapt the floor on (probable) non-speech: fast down, slow up
        if (!voiced) {
            noiseFloorRms = if (rms < noiseFloorRms) {
                0.5 * noiseFloorRms + 0.5 * rms
            } else {
                0.98 * noiseFloorRms + 0.02 * rms
            }
        }
        return voiced
    }

    /** Current adaptive noise floor (RMS) — exposed for debugging/telemetry. */
    fun noiseFloor(): Double = noiseFloorRms
}
