package ai.slashh.runtime

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import ai.slashh.runtime.WhisperConfig as W

/**
 * Runs Whisper on the Hexagon NPU **out-of-process**. On a retail Galaxy S25 the app
 * (`sec_untrusted_app`) is SELinux-blocked from the cDSP, so a shell-domain helper
 * (`whisper_qnn --watch`, uid 2000) drives the QNN encoder/decoder context binaries and
 * exchanges data over a file channel in the app's external files dir. This is the exact same
 * mechanism the WavLM stress rig ([ai.slashh.audio.NpuHelperScorer]) already uses; only the
 * markers differ. No INTERNET, no extra permission — the audio never leaves the device.
 *
 * Protocol (single-flight; app = producer of PCM / consumer of text):
 *   1. app    → writes `whisper_in.raw` (mono float32 LE), clears any stale result, then
 *               touches `whisper_in.ready`.
 *   2. helper → mel → encoder.bin → decoder loop → detokenize → writes `whisper_out.txt`,
 *               touches `whisper_out.ready` (or `whisper_err.ready` on failure).
 *   3. app    → reads the UTF-8 transcript and clears the markers.
 *
 * Honest by construction: if the helper isn't running (no marker within [W.TIMEOUT_MS]) this
 * returns `""` and reports a non-NPU backend — it never invents a transcript.
 */
class NpuWhisperTranscriber(context: Context) : Transcriber {

    private val channel: File = context.getExternalFilesDir(null) ?: context.filesDir

    @Volatile private var lastBackend = Transcriber.Backend.FAKE
    override val backend: Transcriber.Backend get() = lastBackend

    override suspend fun transcribe(samples: FloatArray): String = withContext(Dispatchers.IO) {
        if (samples.isEmpty()) return@withContext ""

        val inRaw = File(channel, W.IN_RAW)
        val inReady = File(channel, W.IN_READY)
        val outTxt = File(channel, W.OUT_TXT)
        val outReady = File(channel, W.OUT_READY)
        val errReady = File(channel, W.ERR_READY)

        // Publish the request: write the payload to a temp file, clear any old result, then
        // atomically rename in and raise the ready flag (so the helper never sees a partial).
        val published = runCatching {
            val staging = File(channel, ".${W.IN_RAW}.tmp")
            staging.outputStream().use { it.write(floatsToLeBytes(samples)) }
            outReady.delete(); outTxt.delete(); errReady.delete()
            // Same dir → renameTo is atomic on Android; copy + clean up only if it somehow didn't
            // move (guard staging.exists() so we never copyTo a temp the rename already consumed).
            if (!staging.renameTo(inRaw) && staging.exists()) {
                staging.copyTo(inRaw, overwrite = true)
                staging.delete()
            }
            inReady.writeBytes(ByteArray(0))
        }.isSuccess
        if (!published) { lastBackend = Transcriber.Backend.FAKE; return@withContext "" }

        // Poll for the helper's result.
        var waited = 0L
        while (waited < W.TIMEOUT_MS) {
            if (errReady.exists()) {
                errReady.delete()
                lastBackend = Transcriber.Backend.FAKE
                return@withContext ""
            }
            if (outReady.exists()) {
                // The channel is FUSE + cross-uid (shell writes, app reads); a just-written file
                // can briefly EACCES for the app while FUSE propagates, so retry the read.
                var text = ""
                for (attempt in 0 until 20) {
                    val r = runCatching { outTxt.readText(Charsets.UTF_8).trim() }
                    if (r.isSuccess) { text = r.getOrDefault(""); break }
                    delay(50)
                }
                outReady.delete(); outTxt.delete()
                lastBackend = Transcriber.Backend.QNN_NPU
                return@withContext text
            }
            delay(W.POLL_MS)
            waited += W.POLL_MS
        }
        // Helper not running / too slow → honest empty, no fabrication.
        lastBackend = Transcriber.Backend.FAKE
        ""
    }

    private fun floatsToLeBytes(f: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(f.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (x in f) bb.putFloat(x)
        return bb.array()
    }
}
