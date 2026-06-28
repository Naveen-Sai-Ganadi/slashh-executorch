package ai.slashh.audio

import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * Raw-waveform stress scorer that runs WavLM stage 2 on the Hexagon NPU **out of
 * process**, over a file channel, because a retail (locked) Galaxy S25 SELinux-blocks
 * the app domain (`sec_untrusted_app`) from `/dev/fastrpc-cdsp`. An in-app QNN
 * delegate ([WavLmStressClassifier]) therefore dies with error 4000. The shell
 * domain (uid 2000) *does* have cDSP access, so a companion `npu_helper.sh` running
 * there drives the HTP on our behalf and we hand audio back and forth through the
 * app's own external files dir (`getExternalFilesDir(null)` — group `ext_data_rw`
 * is shared by shell and the app, so this needs **no** INTERNET and **no** new
 * permission, preserving the on-device-only privacy stance).
 *
 * PROTOCOL (single-flight; this class is the producer of input / consumer of output):
 *   1. clear any stale output markers (npu_out.ready / npu_out.raw / npu_err.ready)
 *   2. write the 192000-byte LE float32 window to `.npu_in.tmp`, rename -> `npu_in.raw`
 *   3. touch `npu_in.ready`
 *   4. poll for `npu_out.ready` (success) or `npu_err.ready` (helper failure)
 *   5. on success read the 4-byte LE float logit from `npu_out.raw`, sigmoid it,
 *      delete the output markers, and return the probability.
 * On timeout or helper error we fall back to [fallback] (if given) or return 0f, so
 * a stalled/absent helper degrades to "no stress" rather than blocking the worker.
 *
 * [scoreWindow] is single-flight by contract — the monitor service calls it serially
 * from one worker thread — and `synchronized` here guards against accidental reentry.
 */
class NpuHelperScorer(
    private val channelDir: File,
    private val fallback: RawWaveScorer? = null,
    private val timeoutMs: Long = 4_000L,
    private val pollMs: Long = 50L,
) : RawWaveScorer {

    private val inTmp = File(channelDir, ".npu_in.tmp")
    private val inRaw = File(channelDir, "npu_in.raw")
    private val inReady = File(channelDir, "npu_in.ready")
    private val outRaw = File(channelDir, "npu_out.raw")
    private val outReady = File(channelDir, "npu_out.ready")
    private val errReady = File(channelDir, "npu_err.ready")

    private val lock = Any()
    private var served = 0L

    override fun scoreWindow(pcm: FloatArray): Float {
        require(pcm.size == AudioConfig.WINDOW_SAMPLES) {
            "pcm length ${pcm.size} != ${AudioConfig.WINDOW_SAMPLES} (WINDOW_SAMPLES)"
        }
        synchronized(lock) {
            val npu = try {
                runOnce(pcm, timeoutMs)
            } catch (e: Throwable) {
                Log.w("Slashh", "NpuHelperScorer exchange failed: ${e.message}")
                null
            }
            return npu ?: fallbackScore(pcm)
        }
    }

    /**
     * One-shot liveness check: send a synthetic window and report whether the helper
     * actually answered on the NPU (vs. timed out / errored). Used at service start
     * to decide whether to route through the NPU at all — and it warms the HTP skel
     * + context so the first real window isn't slow. Uses a longer timeout because
     * the first qnn-net-run loads the 644 MB context cold.
     */
    fun probe(probeTimeoutMs: Long = 15_000L): Boolean {
        synchronized(lock) {
            return try {
                val warm = FloatArray(AudioConfig.WINDOW_SAMPLES) { i -> ((i % 100) - 50) / 500f }
                runOnce(warm, probeTimeoutMs) != null
            } catch (e: Throwable) {
                Log.w("Slashh", "NPU probe failed: ${e.message}")
                false
            }
        }
    }

    /**
     * Run a single file-channel exchange. Returns the sigmoid probability if the
     * NPU helper answered within [budgetMs], or null on helper error / timeout (so
     * the caller can fall back). Caller holds [lock].
     */
    private fun runOnce(pcm: FloatArray, budgetMs: Long): Float? {
        // (1) clear stale output so we can't read a previous request's result.
        outReady.delete(); outRaw.delete(); errReady.delete()

        // (2) write input atomically: full buffer to tmp, then rename. The helper
        // only ever opens npu_in.raw after npu_in.ready appears, but renaming keeps
        // it from ever observing a half-written file even if it polled raw directly.
        val buf = ByteBuffer.allocate(pcm.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (s in pcm) buf.putFloat(s)
        inTmp.writeBytes(buf.array())
        if (!inTmp.renameTo(inRaw)) {
            // rename can fail if a stale inRaw lingers; force it.
            inRaw.delete()
            if (!inTmp.renameTo(inRaw)) {
                inTmp.delete()
                throw IllegalStateException("could not stage npu_in.raw")
            }
        }

        // (3) raise the request flag. The helper claims it (rm) before running.
        inReady.createNewFile()

        // (4) poll for completion or helper-reported error.
        val deadline = System.nanoTime() + budgetMs * 1_000_000L
        while (System.nanoTime() < deadline) {
            if (outReady.exists()) {
                val logit = readLogit()
                outReady.delete(); outRaw.delete()
                served++
                return (1f / (1f + exp(-logit))).coerceIn(0f, 1f)
            }
            if (errReady.exists()) {
                errReady.delete()
                Log.w("Slashh", "NPU helper reported error; falling back")
                return null
            }
            Thread.sleep(pollMs)
        }
        // timeout: leave the request markers — a slow helper may still drain them —
        // but don't block the worker any longer.
        Log.w("Slashh", "NPU helper timed out after ${budgetMs}ms; falling back")
        return null
    }

    /**
     * Read the 4-byte little-endian float32 stress logit the helper published. The channel is on
     * a FUSE mount shared cross-uid (shell writes, this app reads); a just-`mv`d file can briefly
     * `EACCES`/short-read for the app while FUSE propagates the new inode, so retry the open for
     * up to ~2 s before giving up.
     */
    private fun readLogit(): Float {
        var last: Throwable? = null
        repeat(40) {
            try {
                val bytes = outRaw.readBytes()
                if (bytes.size >= 4) {
                    return ByteBuffer.wrap(bytes, 0, 4).order(ByteOrder.LITTLE_ENDIAN).float
                }
            } catch (e: Throwable) {
                last = e
            }
            try { Thread.sleep(50) } catch (_: InterruptedException) { return@repeat }
        }
        throw last ?: IllegalStateException("npu_out.raw unreadable / <4 bytes after retries")
    }

    private fun fallbackScore(pcm: FloatArray): Float =
        try {
            fallback?.scoreWindow(pcm) ?: 0f
        } catch (e: Throwable) {
            Log.w("Slashh", "NpuHelperScorer fallback failed: ${e.message}")
            0f
        }
}
