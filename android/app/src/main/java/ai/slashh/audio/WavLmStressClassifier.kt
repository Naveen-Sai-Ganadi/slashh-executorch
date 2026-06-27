package ai.slashh.audio

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import kotlin.math.exp

/**
 * Raw-waveform stress scorer backed by the fine-tuned **WavLM** teacher, lowered to
 * a QNN/Hexagon-delegated ExecuTorch `.pte` (Path A — re-lowered through
 * ExecuTorch's own 2.37 QNN backend for SM8750 / HTP V79, so `Module.load` can run
 * it on the NPU).
 *
 * Unlike [ExecuTorchStressClassifier] (precomputed log-mel features → tiny
 * StressNet), WavLM ingests the **raw** 16 kHz window: per-window normalization,
 * the encoder, mean-pooling and the stress head all live *inside* the graph
 * (`model/export_teacher.DeployTeacher`). So we feed the `[1,48000]` pcm straight
 * through and apply the sigmoid the graph deliberately leaves off — the head emits
 * a raw logit.
 *
 * The `.pte` is ~hundreds of MB (too large to bundle in the APK): push it to the
 * app's external files dir with
 * `adb push … /sdcard/Android/data/ai.slashh/files/teacher_wavlm_broad_qnn.pte`
 * and construct with that absolute path. [scoreWindow] is synchronized, so the
 * one-shot warmup below and the audio worker can never enter `forward` at once.
 */
class WavLmStressClassifier(modelPath: String) : RawWaveScorer, AutoCloseable {

    private val module: Module = Module.load(modelPath)
    private val shape = longArrayOf(1, AudioConfig.WINDOW_SAMPLES.toLong())   // [1,48000]
    private val lock = Any()

    init {
        // Warm the QNN/HTP delegate off the UI thread. The *first* forward pays a
        // one-time cost — loading the Hexagon skel onto the cDSP over fastrpc and
        // preparing the baked context binary — which would otherwise stutter the
        // first voiced window. Running it now also yields an immediate, deterministic
        // on-NPU execution signal in logcat (a delegated QNN partition has no CPU
        // fallback, so a clean forward here *is* proof the graph ran on the Hexagon).
        Thread {
            try {
                val t0 = System.nanoTime()
                // small bounded sawtooth — non-constant so in-graph normalization has
                // a well-defined (non-zero) std, unlike an all-zero buffer.
                val warm = scoreWindow(FloatArray(AudioConfig.WINDOW_SAMPLES) { i ->
                    ((i % 100) - 50) / 500f
                })
                val ms = (System.nanoTime() - t0) / 1_000_000.0
                Log.i("Slashh", "WavLM warmup: QNN/HTP forward OK in %.1f ms (score=%.4f)".format(ms, warm))
            } catch (e: Throwable) {
                Log.e("Slashh", "WavLM warmup forward FAILED: ${e.message}", e)
            }
        }.apply { isDaemon = true; name = "wavlm-warmup" }.start()
    }

    override fun scoreWindow(pcm: FloatArray): Float {
        require(pcm.size == AudioConfig.WINDOW_SAMPLES) {
            "pcm length ${pcm.size} != ${AudioConfig.WINDOW_SAMPLES} (WINDOW_SAMPLES)"
        }
        synchronized(lock) {
            val input = EValue.from(Tensor.fromBlob(pcm, shape))
            val out = module.forward(input)
            val logit = out[0].toTensor().dataAsFloatArray[0]
            return (1f / (1f + exp(-logit))).coerceIn(0f, 1f)        // logit -> probability
        }
    }

    override fun close() {
        synchronized(lock) { module.destroy() }
    }
}
