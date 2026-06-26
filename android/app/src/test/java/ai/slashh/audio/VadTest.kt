package ai.slashh.audio

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

/** VAD must pass voiced audio and gate silence (energy-efficiency lever). */
class VadTest {

    @Test
    fun silenceIsNotVoiced() {
        val vad = Vad()
        assertFalse(vad.isVoiced(FloatArray(AudioConfig.WINDOW_SAMPLES)))
    }

    @Test
    fun loudHarmonicSignalIsVoiced() {
        val vad = Vad()
        val n = AudioConfig.WINDOW_SAMPLES
        val w = FloatArray(n) {
            val t = it.toDouble() / AudioConfig.SAMPLE_RATE
            0.3f * (sin(2 * Math.PI * 150 * t) +
                    sin(2 * Math.PI * 300 * t) +
                    sin(2 * Math.PI * 450 * t)).toFloat()
        }
        assertTrue(vad.isVoiced(w))
    }

    @Test
    fun veryQuietSignalIsGated() {
        val vad = Vad()
        val n = AudioConfig.WINDOW_SAMPLES
        val w = FloatArray(n) {
            1e-4f * sin(2 * Math.PI * 150 * it / AudioConfig.SAMPLE_RATE).toFloat()
        }
        assertFalse(vad.isVoiced(w))
    }
}
