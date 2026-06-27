package ai.slashh.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationTest {

    @Test fun separableVoicesPutThresholdBetweenThem() {
        val calm = listOf(0.10f, 0.15f, 0.12f, 0.18f, 0.14f, 0.11f)
        val stressed = listOf(0.70f, 0.82f, 0.75f, 0.88f, 0.79f, 0.81f)
        val r = Calibration.compute(calm, stressed)
        assertTrue(r.separable)
        assertTrue("enter above calm", r.enter > 0.18f)
        assertTrue("enter below stressed", r.enter < 0.70f)
        assertTrue("release below enter", r.release < r.enter)
    }

    @Test fun overlappingVoicesFlaggedNotSeparable() {
        val calm = listOf(0.40f, 0.55f, 0.48f, 0.60f)
        val stressed = listOf(0.45f, 0.52f, 0.50f, 0.58f)
        val r = Calibration.compute(calm, stressed)
        assertTrue("overlap -> weak", !r.separable)
    }

    @Test fun thresholdsAreClampedToSaneRange() {
        val r = Calibration.compute(listOf(0.0f, 0.0f), listOf(1f, 1f))
        assertTrue(r.enter in 0.15f..0.90f)
        assertTrue(r.release in 0.10f..r.enter)
    }

    @Test fun averagesAreReported() {
        val r = Calibration.compute(listOf(0.2f, 0.2f), listOf(0.8f, 0.8f))
        assertEquals(0.2f, r.calmAvg, 1e-4f)
        assertEquals(0.8f, r.stressAvg, 1e-4f)
    }
}
