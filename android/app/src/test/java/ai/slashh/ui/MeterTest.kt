package ai.slashh.ui

import ai.slashh.audio.AudioConfig
import ai.slashh.audio.StressPipeline.StressState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM tests for the stress-meter presentation logic (M8). These run on a plain
 * JVM (no Android, no device) and pin down the band/color/percent mapping and —
 * importantly — that the meter never disagrees with the pipeline's hysteresis
 * latch.
 */
class MeterTest {

    private fun state(level: Float?, stressed: Boolean = false) =
        StressState(voiced = level != null, rawScore = level, level = level, stressed = stressed)

    @Test fun nullLevelIsIdleWithNoReading() {
        val m = Meter.from(state(level = null))
        assertFalse(m.hasReading)
        assertEquals(0, m.percent)
        assertEquals(StressBand.IDLE, m.band)
        assertEquals(Meter.COLOR_IDLE, m.argb)
        assertFalse(m.stressed)
    }

    @Test fun lowLevelIsCalmGreen() {
        val m = Meter.from(state(level = 0.10f))
        assertTrue(m.hasReading)
        assertEquals(10, m.percent)
        assertEquals(StressBand.CALM, m.band)
        assertEquals(Meter.COLOR_CALM, m.argb)
    }

    @Test fun midBandUnlatchedIsElevatedAmber() {
        // between RELEASE and STRESS thresholds, not latched -> Elevated
        val mid = (AudioConfig.RELEASE_THRESHOLD + AudioConfig.STRESS_THRESHOLD) / 2f
        val m = Meter.from(state(level = mid, stressed = false))
        assertEquals(StressBand.ELEVATED, m.band)
        assertEquals(Meter.COLOR_ELEVATED, m.argb)
    }

    @Test fun latchedStressedAlwaysReadsHighEvenInHysteresisBand() {
        // level has fallen into the hysteresis band but latch is still on:
        // the meter must show HIGH, never contradict the decision loop.
        val inBand = (AudioConfig.RELEASE_THRESHOLD + AudioConfig.STRESS_THRESHOLD) / 2f
        val m = Meter.from(state(level = inBand, stressed = true))
        assertEquals(StressBand.HIGH, m.band)
        assertEquals(Meter.COLOR_HIGH, m.argb)
        assertTrue(m.stressed)
    }

    @Test fun percentRoundsAndClampsTo0_100() {
        assertEquals(46, Meter.from(state(level = 0.455f)).percent)
        assertEquals(100, Meter.from(state(level = 1.5f)).percent)   // defensive clamp
        assertEquals(0, Meter.from(state(level = -0.2f)).percent)
    }
}
