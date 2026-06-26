package ai.slashh.audio

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * [StressClassifier] backed by the ExecuTorch Android runtime (M7). Loads the
 * `stress_model.pte` produced by `model/export_executorch.py` and runs `forward`
 * on the `[1,1,N_MELS,N_FRAMES]` feature tensor — the exact host path proven by
 * `tests/test_parity.py`, now on device. The XNNPACK `.pte` is the portable
 * default; the QNN/NPU `.pte` from AI Hub drops in with no code change here.
 *
 * Construct with the absolute path to the `.pte` (copy it out of assets first).
 * Not thread-safe — call [score] from a single worker thread.
 */
class ExecuTorchStressClassifier(modelPath: String) : StressClassifier, AutoCloseable {

    private val module: Module = Module.load(modelPath)
    private val shape = longArrayOf(1, 1, AudioConfig.N_MELS.toLong(), AudioConfig.N_FRAMES.toLong())
    private val expected = AudioConfig.N_MELS * AudioConfig.N_FRAMES

    override fun score(features: FloatArray): Float {
        require(features.size == expected) {
            "features length ${features.size} != $expected (N_MELS*N_FRAMES)"
        }
        val input = EValue.from(Tensor.fromBlob(features, shape))
        val out = module.forward(input)
        val v = out[0].toTensor().dataAsFloatArray[0]
        return v.coerceIn(0f, 1f)              // clamp per plan §12
    }

    override fun close() {
        module.destroy()
    }
}
