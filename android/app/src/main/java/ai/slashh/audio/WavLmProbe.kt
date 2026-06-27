package ai.slashh.audio

import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File

/**
 * One-shot on-device smoke test for the INT8 WavLM teacher `.pte` (host-built by
 * scratchpad/export_wavlm_int8_surgical.py). NOT part of the product path — it
 * just proves the 317 MB quantized transformer loads and runs a raw `[1,48000]`
 * 16 kHz waveform through the ExecuTorch runtime on this device, and logs the
 * load/inference latency + the stress logit. Tag: "WavLMProbe".
 *
 * Reads from the app's external files dir (adb-pushable, app-readable; avoids the
 * SELinux block on /data/local/tmp for untrusted_app).
 */
object WavLmProbe {
    private const val TAG = "WavLMProbe"
    private const val N = 48000

    fun run(modelFile: File) {
        if (!modelFile.exists()) {
            Log.w(TAG, "no model at ${modelFile.absolutePath} — skip")
            return
        }
        try {
            Log.i(TAG, "loading ${modelFile.absolutePath} (${modelFile.length() / 1_000_000} MB)")
            val t0 = System.currentTimeMillis()
            val module = Module.load(modelFile.absolutePath)
            Log.i(TAG, "loaded in ${System.currentTimeMillis() - t0} ms")

            // deterministic synthetic 16 kHz window (no mic needed)
            val wave = FloatArray(N) { (Math.sin(it * 0.013) * 0.05).toFloat() }
            val input = EValue.from(Tensor.fromBlob(wave, longArrayOf(1, N.toLong())))

            module.forward(input)  // warmup (lazy alloc / first-run paths)
            val tInf = System.currentTimeMillis()
            val out = module.forward(input)
            val ms = System.currentTimeMillis() - tInf
            val logit = out[0].toTensor().dataAsFloatArray[0]
            Log.i(TAG, "INT8 WavLM ran on-device: logit=$logit inference=${ms}ms " +
                "decision=${if (logit > 0f) "STRESSED" else "CALM"}")
            module.destroy()
            Log.i(TAG, "PROBE_OK")
        } catch (e: Throwable) {
            Log.e(TAG, "PROBE_FAIL ${e.javaClass.simpleName}: ${e.message}", e)
        }
    }
}
