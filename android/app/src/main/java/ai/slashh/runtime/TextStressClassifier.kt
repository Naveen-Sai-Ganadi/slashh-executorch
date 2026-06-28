package ai.slashh.runtime

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * Stress classifier over a transcript. Loads `text_stress.pte` (Linear→ReLU→Linear→sigmoid over
 * an `[1,V]` L2-normalized bag-of-words) plus the matching `text_stress_vocab.txt`, and returns
 * a stress probability in `[0,1]` — or null when the transcript carries no in-vocab signal, so
 * the fusion can ignore an empty/irrelevant transcript rather than pull toward the model's bias.
 *
 * Uses the exact same ExecuTorch runtime path as [ai.slashh.audio.ExecuTorchStressClassifier]
 * (Module.load → Tensor.fromBlob → forward). Construct with the absolute `.pte` path (copy from
 * assets first) and the vocab lines. Not thread-safe — call [score] from one worker thread.
 */
class TextStressClassifier(
    modelPath: String,
    vocabLines: List<String>,
) : AutoCloseable {
    private val module: Module = Module.load(modelPath)
    private val vocab: Map<String, Int> = TextFeatures.loadVocab(vocabLines)
    private val vocabSize: Int = vocabLines.size
    private val shape = longArrayOf(1, vocabSize.toLong())

    /** @return stress prob in [0,1], or null if the transcript has no in-vocab tokens. */
    fun score(transcript: String): Float? {
        val vec = TextFeatures.vectorize(transcript, vocab, vocabSize) ?: return null
        val out = module.forward(EValue.from(Tensor.fromBlob(vec, shape)))
        return out[0].toTensor().dataAsFloatArray[0].coerceIn(0f, 1f)
    }

    override fun close() {
        module.destroy()
    }
}
