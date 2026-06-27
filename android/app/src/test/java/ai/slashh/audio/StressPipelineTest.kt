package ai.slashh.audio

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

/**
 * Pipeline logic: VAD gating, and the energy-driven stress level + hysteresis.
 * The level now tracks vocal intensity (RMS), not the classifier score (which
 * saturates on real on-device speech), so tests vary the window amplitude.
 */
class StressPipelineTest {

    /** Voiced tone at a given amplitude — amplitude controls RMS → stress level. */
    private fun voicedWindow(amp: Float = 0.3f): FloatArray {
        val n = AudioConfig.WINDOW_SAMPLES
        return FloatArray(n) {
            val t = it.toDouble() / AudioConfig.SAMPLE_RATE
            amp * (sin(2 * Math.PI * 150 * t) +
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
    fun loudVoiceReadsHighStress() {
        val pipe = StressPipeline(classifier = { 0.5f })   // classifier irrelevant to level
        val s = pipe.onWindow(voicedWindow(0.3f))          // loud/intense
        assertTrue(s.voiced)
        assertTrue("loud voice -> high stress", s.level!! > 0.7f)
    }

    @Test
    fun hysteresisLatchesOnLoudReleasesOnQuiet() {
        val pipe = StressPipeline(classifier = { 0.5f })
        var s = pipe.onWindow(voicedWindow(0.3f))
        repeat(4) { s = pipe.onWindow(voicedWindow(0.3f)) }
        assertTrue("latches stressed while loud", s.stressed)

        // quiet-but-voiced windows pull the EMA below the release threshold
        repeat(8) { s = pipe.onWindow(voicedWindow(0.03f)) }
        assertFalse("clears once quiet", s.stressed)
    }

    @Test
    fun resetClearsState() {
        val pipe = StressPipeline(classifier = { 0.95f })
        repeat(6) { pipe.onWindow(voicedWindow(0.3f)) }
        pipe.reset()
        val s = pipe.onWindow(silentWindow())
        assertNull(s.level)
        assertFalse(s.stressed)
    }
}
