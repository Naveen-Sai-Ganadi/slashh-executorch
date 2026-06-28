package ai.slashh.runtime

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Feature-parity gate for the text stress classifier: the Kotlin [TextFeatures] tokenizer +
 * vectorizer must reproduce the Python `model/text_features.py` exactly, or the shipped
 * `text_stress.pte` sees different inputs on-device than it trained on. Runs on the JVM (no
 * native ExecuTorch, no torch) — it validates tokenization + the L2-normalized bag-of-words,
 * not the `.pte` forward (that eager↔pte parity is checked in Python and on-device).
 *
 * Regenerate the golden + vocab with `python -m model.text_stress`. The test skips cleanly if
 * the artifacts aren't present yet (so the suite stays green before the models are built).
 */
class TextFeaturesParityTest {

    private fun goldenOrNull(): JSONObject? {
        val stream = javaClass.classLoader?.getResourceAsStream("golden_text_fusion.json")
            ?: return null
        return JSONObject(stream.bufferedReader().use { it.readText() })
    }

    /** The shipped vocab lives in main assets; unit tests run with cwd = module dir. */
    private fun vocabLinesOrNull(): List<String>? {
        val candidates = listOf(
            "src/main/assets/text_stress_vocab.txt",
            "android/app/src/main/assets/text_stress_vocab.txt",
        )
        val f = candidates.map { File(it) }.firstOrNull { it.exists() } ?: return null
        return f.readLines().map { it.trim() }.filter { it.isNotEmpty() }
    }

    @Test
    fun vocabSizeMatchesGolden() {
        val golden = goldenOrNull() ?: run { assumeTrue("golden not built yet", false); return }
        val vocab = vocabLinesOrNull() ?: run { assumeTrue("vocab not built yet", false); return }
        assertEquals("vocab size != golden vocab_size", golden.getInt("vocab_size"), vocab.size)
    }

    @Test
    fun tokenizeAndVectorizeMatchPython() {
        val golden = goldenOrNull() ?: run { assumeTrue("golden not built yet", false); return }
        val vocabLines = vocabLinesOrNull() ?: run { assumeTrue("vocab not built yet", false); return }
        val vocab = TextFeatures.loadVocab(vocabLines)
        val vocabSize = vocabLines.size

        val cases = golden.getJSONArray("text_cases")
        for (c in 0 until cases.length()) {
            val case = cases.getJSONObject(c)
            val text = case.getString("text")

            // 1) tokenization parity
            val expectTokens = case.getJSONArray("tokens")
            val actualTokens = TextFeatures.tokenize(text)
            assertEquals("token count for '${text.take(30)}'", expectTokens.length(), actualTokens.size)
            for (i in 0 until expectTokens.length()) {
                assertEquals("token[$i] for '${text.take(30)}'", expectTokens.getString(i), actualTokens[i])
            }

            // 2) vectorization parity (sparse, against the real shipped vocab)
            val expectNonzero = case.getJSONObject("nonzero")
            val vec = TextFeatures.vectorize(text, vocab, vocabSize)
            if (expectNonzero.length() == 0) {
                assertTrue("expected null/empty vector for '${text.take(30)}'", vec == null)
                continue
            }
            assertTrue("expected non-null vector for '${text.take(30)}'", vec != null)
            // every expected nonzero index matches within tolerance
            val keys = expectNonzero.keys()
            var nonzeroSeen = 0
            while (keys.hasNext()) {
                val k = keys.next()
                val idx = k.toInt()
                val e = expectNonzero.getDouble(k)
                assertTrue("index $idx out of range", idx in 0 until vocabSize)
                assertEquals("vec[$idx] for '${text.take(30)}'", e, vec!![idx].toDouble(), 1e-4)
                nonzeroSeen++
            }
            // and no extra nonzeros beyond what's expected
            var actualNonzero = 0
            for (v in vec!!) if (abs(v) > 1e-7f) actualNonzero++
            assertEquals("nonzero count for '${text.take(30)}'", nonzeroSeen, actualNonzero)
        }
    }
}
