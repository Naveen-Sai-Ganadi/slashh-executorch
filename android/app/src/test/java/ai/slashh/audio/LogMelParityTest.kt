package ai.slashh.audio

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * M6 acceptance gate: the native [LogMel] extractor reproduces the torchaudio
 * features in `model/features.py` within 1e-3 max-abs-error on shared golden
 * vectors. Runs on the JVM (no device, no torchaudio) — regenerate the resource
 * with `python -m model.golden` whenever the feature pipeline changes.
 */
class LogMelParityTest {

    private fun loadGolden(): JSONObject {
        val stream = javaClass.classLoader!!.getResourceAsStream("golden_logmel.json")
            ?: error("golden_logmel.json not on the test classpath — run `python -m model.golden`")
        return JSONObject(stream.bufferedReader().use { it.readText() })
    }

    @Test
    fun configMatchesPython() {
        val cfg = loadGolden().getJSONObject("config")
        assertEquals(AudioConfig.SAMPLE_RATE, cfg.getInt("sample_rate"))
        assertEquals(AudioConfig.N_FFT, cfg.getInt("n_fft"))
        assertEquals(AudioConfig.HOP_LENGTH, cfg.getInt("hop_length"))
        assertEquals(AudioConfig.N_MELS, cfg.getInt("n_mels"))
        assertEquals(AudioConfig.N_FRAMES, cfg.getInt("n_frames"))
        assertEquals(AudioConfig.WINDOW_SAMPLES, cfg.getInt("window_samples"))
    }

    @Test
    fun logMelMatchesTorchaudioWithinTolerance() {
        val golden = loadGolden()
        val cases = golden.getJSONArray("cases")
        val logMel = LogMel()

        for (c in 0 until cases.length()) {
            val case = cases.getJSONObject(c)
            val name = case.getString("name")
            val tol = case.getDouble("tolerance")

            val pcmJson = case.getJSONArray("pcm")
            val pcm = FloatArray(pcmJson.length()) { pcmJson.getDouble(it).toFloat() }

            val expected = case.getJSONArray("logmel") // [N_MELS][N_FRAMES]
            val actual = logMel.extract(pcm)

            var maxErr = 0.0
            for (m in 0 until AudioConfig.N_MELS) {
                val row = expected.getJSONArray(m)
                for (i in 0 until AudioConfig.N_FRAMES) {
                    val e = row.getDouble(i)
                    maxErr = maxOf(maxErr, abs(e - actual[m][i]).toDouble())
                }
            }
            assertTrue(
                "log-mel parity for '$name' failed: max abs err $maxErr >= $tol",
                maxErr < tol
            )
        }
    }
}
