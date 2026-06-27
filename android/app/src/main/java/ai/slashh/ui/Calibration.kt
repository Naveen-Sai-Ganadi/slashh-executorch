package ai.slashh.ui

import kotlin.math.abs

/**
 * Per-user calibration. The model's raw score is offset and (on-device) often
 * INVERTED — calm reads high, stressed low. So instead of a threshold, we learn
 * two anchors: the user's mean CALM raw score and mean STRESSED raw score. The
 * pipeline maps raw -> [0,1] stress between them, which fixes both the offset and
 * the polarity automatically. Pure + android-free so it's unit-testable.
 */
object Calibration {

    data class Result(
        val calmAnchor: Float,    // raw score that should map to 0% stress
        val stressAnchor: Float,  // raw score that should map to 100% stress
        /** false when calm/stressed are too close — re-record with a bigger contrast */
        val separable: Boolean,
    )

    private const val MIN_SEPARATION = 0.03f   // energy units (RMS)

    fun compute(calm: List<Float>, stressed: List<Float>): Result {
        require(calm.isNotEmpty() && stressed.isNotEmpty()) { "need calm and stressed samples" }
        val calmAvg = calm.average().toFloat()
        val stressAvg = stressed.average().toFloat()
        // separable regardless of direction (the model may be inverted)
        val separable = abs(stressAvg - calmAvg) >= MIN_SEPARATION
        return Result(calmAvg, stressAvg, separable)
    }
}
