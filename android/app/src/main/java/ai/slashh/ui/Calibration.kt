package ai.slashh.ui

import kotlin.math.abs

/**
 * Per-user calibration with auto-select. Records the model's score AND vocal
 * intensity (energy) while the user speaks calmly vs stressed, then picks
 * whichever signal actually separates THIS voice:
 *  - the retrained model (genuine ML stress) if it clearly separates, else
 *  - vocal intensity (energy), the robust fallback.
 * Pure + android-free so it's unit-testable.
 */
object Calibration {

    data class Result(
        val useModel: Boolean,
        val calmAnchor: Float,    // chosen-signal value that maps to 0% stress
        val stressAnchor: Float,  // chosen-signal value that maps to 100% stress
        val separable: Boolean,   // false -> re-record with a bigger contrast
    )

    // The model clearly separates this voice if calm vs stressed means differ by
    // this much (model output is 0..1). Below it, energy is more reliable.
    private const val MODEL_SEPARATION = 0.25f
    private const val MODEL_MIN = 0.15f
    private const val ENERGY_MIN = 0.03f

    fun compute(
        calmRaw: List<Float>, calmEnergy: List<Float>,
        stressRaw: List<Float>, stressEnergy: List<Float>,
    ): Result {
        require(calmRaw.isNotEmpty() && stressRaw.isNotEmpty()) { "need calm and stressed samples" }
        val mCalm = calmRaw.average().toFloat()
        val mStress = stressRaw.average().toFloat()
        val eCalm = calmEnergy.average().toFloat()
        val eStress = stressEnergy.average().toFloat()

        val modelSep = abs(mStress - mCalm)
        val useModel = modelSep >= MODEL_SEPARATION
        return if (useModel) {
            Result(true, mCalm, mStress, modelSep >= MODEL_MIN)
        } else {
            Result(false, eCalm, eStress, abs(eStress - eCalm) >= ENERGY_MIN)
        }
    }
}
