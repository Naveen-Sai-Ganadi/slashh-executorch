package ai.slashh.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

/** Pipeline logic: VAD gating, EMA smoothing, and hysteresis (M6 + M8). */
class StressPipelineTest {

    // a loud, harmonic-rich window so the real VAD passes it (ZCR ~0.056, in band)
    private fun voicedWindow(): FloatArray {
        val n = AudioConfig.WINDOW_SAMPLES
        return FloatArray(n) {
            val t = it.toDouble() / AudioConfig.SAMPLE_RATE
            0.3f * (sin(2 * Math.PI * 150 * t) +
                    sin(2 * Math.PI * 300 * t) +
                    sin(2 * Math.PI * 450 * t)).toFloat()
        }
    }

    private fun silentWindow() = FloatArray(AudioConfig.WINDOW_SAMPLES)

    @Test
    fun silenceIsGatedAndYieldsNoLevel() {
        val pipe = StressPipeline(classifier = { 0.9f })
        val s = pipe.onWindow(silentWindow())
        assertFalse(s.voiced)
        assertNull("no level before any voiced window", s.level)
        assertNull(s.rawScore)
        assertFalse(s.stressed)
    }

    @Test
    fun hysteresisLatchesAndReleasesAtSeparateThresholds() {
        // score we control: high then low
        var score = 0.95f
        val pipe = StressPipeline(classifier = { score })

        // drive several high windows → EMA climbs past STRESS_THRESHOLD
        var s = pipe.onWindow(voicedWindow())
        repeat(5) { s = pipe.onWindow(voicedWindow()) }
        assertTrue("should latch stressed once EMA >= ${AudioConfig.STRESS_THRESHOLD}", s.stressed)

        // a value BETWEEN release and stress thresholds must NOT clear the latch
        score = 0.5f   // 0.45 < 0.5 < 0.6
        repeat(6) { s = pipe.onWindow(voicedWindow()) }
        assertTrue("stays stressed in the hysteresis band", s.stressed)

        // drop below RELEASE_THRESHOLD → clears
        score = 0.1f
        repeat(6) { s = pipe.onWindow(voicedWindow()) }
        assertFalse("clears once EMA < ${AudioConfig.RELEASE_THRESHOLD}", s.stressed)
    }

    @Test
    fun firstVoicedScoreSeedsEmaExactly() {
        val pipe = StressPipeline(classifier = { 0.42f })
        val s = pipe.onWindow(voicedWindow())
        assertTrue(s.voiced)
        assertEquals(0.42f, s.level!!, 1e-6f)   // EMA seeds to the first raw score
    }

    @Test
    fun resetClearsState() {
        val pipe = StressPipeline(classifier = { 0.95f })
        repeat(6) { pipe.onWindow(voicedWindow()) }
        pipe.reset()
        val s = pipe.onWindow(silentWindow())
        assertNull(s.level)
        assertFalse(s.stressed)
    }
}
