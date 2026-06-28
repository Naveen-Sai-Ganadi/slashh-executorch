package ai.slashh.runtime

import kotlin.math.sqrt

/**
 * Shared, deterministic text featurization for the stress text classifier — the Kotlin mirror
 * of `model/text_features.py`. This MUST stay byte-for-byte equivalent to the Python side, or
 * the trained `text_stress.pte` would see different vectors on-device than it trained on (the
 * same single-source-of-truth discipline as [ai.slashh.audio.AudioConfig] vs `audio_config.py`).
 *
 * Contract:
 *   tokenize  : lowercase → replace every char not in `a-z` with a space → split on whitespace,
 *               dropping empties. Unigrams only; no stemming, stopwords, or n-grams.
 *   vectorize : L2-normalized term-count vector over the fixed vocab (token → index).
 */
object TextFeatures {
    private val NON_ALPHA = Regex("[^a-z]+")

    /** Tokenize per the shared contract. */
    fun tokenize(text: String): List<String> =
        text.lowercase().replace(NON_ALPHA, " ").split(' ').filter { it.isNotEmpty() }

    /**
     * Build the token→index map from vocab lines (line i → index i). Drops blank lines
     * defensively; the shipped `text_stress_vocab.txt` has exactly one token per line.
     */
    fun loadVocab(lines: List<String>): Map<String, Int> {
        val m = HashMap<String, Int>(lines.size * 2)
        for ((i, raw) in lines.withIndex()) {
            val tok = raw.trim()
            if (tok.isNotEmpty()) m[tok] = i
        }
        return m
    }

    /**
     * L2-normalized term-count vector, shape [vocabSize]. Returns null when no token is in
     * vocab (norm 0) — there is no learned signal to fuse, so the caller should ignore text.
     */
    fun vectorize(text: String, vocab: Map<String, Int>, vocabSize: Int): FloatArray? {
        val counts = FloatArray(vocabSize)
        var any = false
        for (tok in tokenize(text)) {
            val idx = vocab[tok] ?: continue
            counts[idx] += 1f
            any = true
        }
        if (!any) return null
        var ss = 0.0
        for (c in counts) ss += (c.toDouble() * c.toDouble())
        val norm = sqrt(ss).toFloat()
        if (norm <= 0f) return null
        for (i in counts.indices) counts[i] = counts[i] / norm
        return counts
    }
}
