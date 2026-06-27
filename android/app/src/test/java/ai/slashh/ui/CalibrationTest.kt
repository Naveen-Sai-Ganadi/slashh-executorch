package ai.slashh.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationTest {

    @Test fun anchorsAreTheMeanScores() {
        val r = Calibration.compute(listOf(0.90f, 0.94f), listOf(0.50f, 0.48f))
        assertEquals(0.92f, r.calmAnchor, 1e-3f)
        assertEquals(0.49f, r.stressAnchor, 1e-3f)
    }

    @Test fun invertedModelStillSeparable() {
        // on-device the model is inverted: calm reads HIGH, stressed LOW
        val r = Calibration.compute(listOf(0.95f, 0.97f), listOf(0.50f, 0.55f))
        assertTrue(r.separable)
        assertTrue("calm anchor high", r.calmAnchor > r.stressAnchor)
    }

    @Test fun normalDirectionAlsoSeparable() {
        val r = Calibration.compute(listOf(0.10f, 0.15f), listOf(0.80f, 0.85f))
        assertTrue(r.separable)
        assertTrue(r.stressAnchor > r.calmAnchor)
    }

    @Test fun tooCloseFlaggedNotSeparable() {
        val r = Calibration.compute(listOf(0.50f, 0.55f), listOf(0.52f, 0.57f))
        assertTrue(!r.separable)
    }
}
