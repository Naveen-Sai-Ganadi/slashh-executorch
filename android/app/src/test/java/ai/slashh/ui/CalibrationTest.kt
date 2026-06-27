package ai.slashh.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CalibrationTest {

    @Test fun picksModelWhenItClearlySeparates() {
        val r = Calibration.compute(
            calmRaw = listOf(0.20f, 0.25f), calmEnergy = listOf(0.05f, 0.06f),
            stressRaw = listOf(0.85f, 0.90f), stressEnergy = listOf(0.07f, 0.08f),
        )
        assertTrue(r.useModel)
        assertTrue(r.separable)
        assertEquals(0.225f, r.calmAnchor, 1e-2f)   // model raw means
    }

    @Test fun fallsBackToEnergyWhenModelSaturates() {
        // model reads ~0.96 for both (saturated) — energy must win
        val r = Calibration.compute(
            calmRaw = listOf(0.95f, 0.96f), calmEnergy = listOf(0.03f, 0.04f),
            stressRaw = listOf(0.97f, 0.98f), stressEnergy = listOf(0.13f, 0.14f),
        )
        assertTrue(!r.useModel)
        assertTrue(r.separable)
        assertEquals(0.035f, r.calmAnchor, 1e-2f)   // energy means
    }

    @Test fun invertedModelStillCountsAsSeparation() {
        val r = Calibration.compute(
            calmRaw = listOf(0.95f), calmEnergy = listOf(0.03f),
            stressRaw = listOf(0.50f), stressEnergy = listOf(0.13f),
        )
        assertTrue(r.useModel)                       // |0.50-0.95| = 0.45 >= 0.25
        assertTrue("inverted: calm anchor high", r.calmAnchor > r.stressAnchor)
    }

    @Test fun neitherSeparatesFlagged() {
        val r = Calibration.compute(
            calmRaw = listOf(0.95f), calmEnergy = listOf(0.05f),
            stressRaw = listOf(0.97f), stressEnergy = listOf(0.06f),
        )
        assertTrue(!r.separable)
    }
}
