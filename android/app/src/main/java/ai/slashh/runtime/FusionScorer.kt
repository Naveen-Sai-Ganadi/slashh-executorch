package ai.slashh.runtime

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * Late-fusion perceptron: combines the audio model's per-window stress score with the latest
 * text-stress score into a single, more confident probability. Loads `fusion.pte`
 * (Linear→ReLU→Linear→sigmoid, input `[1,2]` = `[audio, text]`, output `[1,1]`). Trained so the
 * fused score is more confident than either signal alone when they agree, and hedges when they
 * disagree.
 *
 * When no fresh text score is available the caller should use the audio score directly rather
 * than calling this (the fusion was trained on genuine paired signals, not a constant filler).
 * Not thread-safe — call [fuse] from one worker thread.
 */
class FusionScorer(modelPath: String) : AutoCloseable {
    private val module: Module = Module.load(modelPath)
    private val shape = longArrayOf(1, 2)

    /** @return fused stress prob in [0,1]. */
    fun fuse(audioScore: Float, textScore: Float): Float {
        val input = floatArrayOf(audioScore.coerceIn(0f, 1f), textScore.coerceIn(0f, 1f))
        val out = module.forward(EValue.from(Tensor.fromBlob(input, shape)))
        return out[0].toTensor().dataAsFloatArray[0].coerceIn(0f, 1f)
    }

    override fun close() {
        module.destroy()
    }
}
