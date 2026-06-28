package ai.slashh.audio

import android.util.Log
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * Runs the WavLM audio-stress model on the Hexagon NPU **asynchronously**, decoupled from the
 * fast per-window meter. Each NPU forward (qnn-net-run reloads the 614 MB context) takes
 * ~6–11 s — far slower than the 1 s hop — so scoring it synchronously inside the pipeline would
 * freeze the meter and always trip the per-window timeout. Instead a dedicated worker scores the
 * latest captured window on its own cadence and publishes the freshest score, which the pipeline
 * fuses in like the Whisper text score. This is the same async pattern as
 * [ai.slashh.runtime.TranscriptionCoordinator], so BOTH NPU models (WavLM audio + Whisper text)
 * run on the Hexagon without blocking the responsive energy meter.
 *
 * @param scorer the out-of-process NPU scorer (a [NpuHelperScorer] with a long per-window
 *   timeout so the ~8 s WavLM forward isn't dropped).
 */
class WavLmCoordinator(
    private val scorer: RawWaveScorer,
    private val freshnessMs: Long = 30_000L,
    /** Minimum gap between NPU forwards. Each WavLM call reloads a 614 MB context (qnn-net-run
     *  is not resident), so this caps the alloc/free churn. 15 s was a memory-safety valve when
     *  the device had <1 GB free; with headroom (post-reboot, ~5-7 GB) ~2 s gives a responsive
     *  update without back-to-back 614 MB reloads. The per-call latency (~1-3 s) is the floor. */
    private val minIntervalMs: Long = 2_000L,
    /** EMA factor for the published score — each NPU call scores a different audio window, so
     *  raw scores jump (a silent window vs a speech window); smoothing tracks recent audio. */
    private val emaAlpha: Float = 0.5f,
) {
    private val latest = AtomicReference<FloatArray?>(null)
    @Volatile private var worker: Thread? = null
    @Volatile private var running = false
    @Volatile private var scoreRaw: Float? = null   // EMA-smoothed, published
    @Volatile private var lastAtMs = 0L
    private var ema = Float.NaN

    /** Stash the newest window (latest-wins; the worker scores whatever is current). */
    fun onWindow(window: FloatArray) = latest.set(window)

    /** Latest WavLM stress score in [0,1], or null if none / stale. */
    fun score(nowMs: Long = System.currentTimeMillis()): Float? {
        val s = scoreRaw ?: return null
        return if (nowMs - lastAtMs <= freshnessMs) s else null
    }

    fun start() {
        if (running) return
        running = true
        worker = thread(name = "slashh-wavlm", isDaemon = true) {
            var lastStart = 0L
            while (running) {
                // Throttle: never start a new 614 MB forward until minIntervalMs has elapsed.
                val sinceStart = System.currentTimeMillis() - lastStart
                if (sinceStart < minIntervalMs) {
                    try { Thread.sleep(minOf(minIntervalMs - sinceStart, 500L)) }
                    catch (_: InterruptedException) { break }
                    continue
                }
                val w = latest.getAndSet(null)
                if (w == null) {
                    try { Thread.sleep(200) } catch (_: InterruptedException) { break }
                    continue
                }
                lastStart = System.currentTimeMillis()
                // NpuHelperScorer returns 0f on timeout/failure (its fallback is null); a real
                // WavLM sigmoid is never exactly 0, so treat 0f as "no score this round".
                val s = try {
                    scorer.scoreWindow(w)
                } catch (e: Throwable) {
                    Log.w("Slashh", "wavlm score failed: ${e.message}"); 0f
                }
                if (s > 0f) {
                    ema = if (ema.isNaN()) s else emaAlpha * s + (1 - emaAlpha) * ema
                    scoreRaw = ema
                    lastAtMs = System.currentTimeMillis()
                    Log.d("Slashh", "wavlm[NPU] raw=$s smoothed=$ema")
                }
            }
        }
    }

    fun stop() {
        running = false
        worker?.interrupt()
        worker = null
    }
}
