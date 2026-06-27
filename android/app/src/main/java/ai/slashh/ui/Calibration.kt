package ai.slashh.ui

import kotlin.math.roundToInt

/**
 * Per-user threshold calibration (M-calib). Given the model's scores while the
 * user spoke *calmly* vs *stressed*, pick an enter/exit threshold that separates
 * them for THIS voice on THIS device. Pure + android-free so it's unit-testable.
 *
 * Strategy: put the enter threshold between the top of the calm scores (80th pct)
 * and the bottom of the stressed scores (20th pct). If the two overlap (the model
 * couldn't separate this person's calm vs stressed), fall back to the midpoint of
 * the means and flag it as weak so the UI can advise a stronger contrast.
 */
object Calibration {

    data class Result(
        val enter: Float,
        val release: Float,
        val calmAvg: Float,
        val stressAvg: Float,
        /** false when calm/stressed overlap — calibration is weak, re-record louder */
        val separable: Boolean,
    )

    fun compute(calm: List<Float>, stressed: List<Float>): Result {
        require(calm.isNotEmpty() && stressed.isNotEmpty()) { "need calm and stressed samples" }
        val calmHi = percentile(calm, 0.80f)
        val strLo = percentile(stressed, 0.20f)
        val calmAvg = calm.average().toFloat()
        val strAvg = stressed.average().toFloat()

        val separable = strAvg > calmAvg && strLo > calmHi
        val enterRaw = if (separable) (calmHi + strLo) / 2f else (calmAvg + strAvg) / 2f
        val enter = enterRaw.coerceIn(0.15f, 0.90f)
        val release = (enter - 0.12f).coerceIn(0.10f, enter - 0.03f)
        return Result(enter, release, calmAvg, strAvg, separable)
    }

    private fun percentile(xs: List<Float>, p: Float): Float {
        val s = xs.sorted()
        val idx = (p * (s.size - 1)).roundToInt().coerceIn(0, s.size - 1)
        return s[idx]
    }
}
