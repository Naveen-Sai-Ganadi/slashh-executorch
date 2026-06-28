package ai.slashh.runtime

import ai.slashh.audio.AudioConfig
import android.util.Log
import kotlinx.coroutines.runBlocking
import kotlin.concurrent.thread

/**
 * Drives Whisper transcription on a slow background cadence, decoupled from the per-window
 * stress scoring. The audio-scoring worker pushes each capture window via [onWindow]; a
 * dedicated worker periodically transcribes the accumulated buffer and (optionally) scores the
 * transcript for stress. The latest transcript + text-stress score are published for the
 * pipeline to fuse with the audio model's per-window score.
 *
 * Transcription is far slower than the 1 s hop (an NPU encode + autoregressive decode loop), so
 * it runs on its own thread and never blocks audio capture or stress scoring. When the helper
 * is down, [Transcriber.transcribe] returns "" and we publish no text signal — the fusion then
 * falls back to the audio score alone (honest degradation).
 *
 * @param textScorer maps a transcript to a stress score in [0,1], or null if no text model is
 *   loaded; may itself return null for an empty/uninformative transcript.
 */
class TranscriptionCoordinator(
    private val transcriber: Transcriber,
    private val textScorer: ((String) -> Float?)? = null,
    bufferSeconds: Double = 10.0,
    private val cadenceMs: Long = 4_000L,
    minSeconds: Double = 3.0,
    private val freshnessMs: Long = 20_000L,
) {
    private val buffer = WhisperBuffer(bufferSeconds)
    private val minSamples = (minSeconds * AudioConfig.SAMPLE_RATE).toInt()

    @Volatile private var worker: Thread? = null
    @Volatile private var running = false

    /** Most recent non-empty transcript (for the UI / honest stats). */
    @Volatile var transcript: String = ""
        private set
    @Volatile private var textScoreRaw: Float? = null
    @Volatile private var lastTextAtMs: Long = 0L
    /** Backend that serviced the last transcription (NPU vs none), for the UI badge. */
    @Volatile var backend: Transcriber.Backend = Transcriber.Backend.FAKE
        private set

    /** Feed each capture window (only its newest hop is retained). */
    fun onWindow(window: FloatArray) = buffer.appendWindow(window)

    /** Latest text-stress score, or null if none / stale / no text model. */
    fun textScore(nowMs: Long = System.currentTimeMillis()): Float? {
        val s = textScoreRaw ?: return null
        return if (nowMs - lastTextAtMs <= freshnessMs) s else null
    }

    fun start() {
        if (running) return
        running = true
        worker = thread(name = "slashh-whisper", isDaemon = true) {
            while (running) {
                try { Thread.sleep(cadenceMs) } catch (_: InterruptedException) { break }
                if (!running) break
                if (buffer.available() < minSamples) continue
                val pcm = buffer.snapshot()
                val text = try {
                    runBlocking { transcriber.transcribe(pcm) }
                } catch (e: Throwable) {
                    Log.w("Slashh", "whisper transcribe failed: ${e.message}")
                    ""
                }
                backend = transcriber.backend
                if (isMeaningful(text)) {
                    transcript = text
                    val sc = try {
                        textScorer?.invoke(text)
                    } catch (e: Throwable) {
                        Log.w("Slashh", "text scorer failed: ${e.message}"); null
                    }
                    if (sc != null) {
                        textScoreRaw = sc
                        lastTextAtMs = System.currentTimeMillis()
                    }
                    Log.d("Slashh", "whisper[$backend] -> \"$text\" textScore=$sc")
                }
            }
        }
    }

    fun stop() {
        running = false
        worker?.interrupt()
        worker = null
    }

    /**
     * Whisper emits placeholder tokens on silence/noise — "[BLANK_AUDIO]", "[ Silence ]",
     * "(music)", etc. Strip bracketed/parenthesized segments and require at least a couple of
     * real letters before treating the window as spoken text, so silence never produces a
     * spurious text-stress score.
     */
    private fun isMeaningful(text: String): Boolean {
        if (text.isBlank()) return false
        val cleaned = text
            .replace(Regex("[\\[(][^\\])]*[\\])]"), " ")   // drop [..] and (..) segments
            .replace(Regex("[^A-Za-z']+"), " ")
            .trim()
        return cleaned.length >= 2
    }
}
