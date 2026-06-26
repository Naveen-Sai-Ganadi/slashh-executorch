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
    private val absFloorRms: Double = 3e-4,     // ~-70 dBFS: never treat quieter than this as voice
) {
    // Seeded to a quiet-room PRIOR (~-60 dBFS), NOT to the first window. Seeding
    // from window #1 would make `threshold = rms*margin > rms`, so the first
    // window could never be voiced — a user who speaks immediately would be
    // missed until the next window. The floor adapts from here (fast down / slow
    // up), so it still tracks the real ambient level within a window or two.
    private var noiseFloorRms = 1e-3

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

        // Compare against the floor, but never below an absolute minimum, so a
        // long silence can't drag the floor to ~0 and then admit faint hum.
        val floor = noiseFloorRms.coerceAtLeast(absFloorRms)
        val threshold = floor * Math.pow(10.0, energyMarginDb / 20.0)
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
