package ai.slashh.runtime

import ai.slashh.audio.AudioConfig

/**
 * Rolling PCM accumulator that reconstructs a continuous [bufferSeconds]-second window from the
 * overlapping per-hop windows [ai.slashh.audio.AudioCapture] emits. Each capture window is the
 * most-recent [AudioConfig.WINDOW_SAMPLES] samples; only its newest [AudioConfig.HOP_SAMPLES]
 * (the last hop) is genuinely new, so we append just that slice and keep the last
 * [bufferSeconds] seconds for Whisper — which needs more than the 3 s stress window to produce
 * a useful transcript.
 *
 * Thread-safe: [appendWindow] is called from the capture/scoring worker, [snapshot] from the
 * transcription worker.
 */
class WhisperBuffer(bufferSeconds: Double = 10.0) {
    private val capacity = (bufferSeconds * AudioConfig.SAMPLE_RATE).toInt()
    private val ring = FloatArray(capacity)
    private var writePos = 0
    private var filled = 0
    private val lock = Any()

    /** Append the newest hop of a capture window (its last [AudioConfig.HOP_SAMPLES] samples). */
    fun appendWindow(window: FloatArray) {
        val hop = AudioConfig.HOP_SAMPLES
        val start = maxOf(0, window.size - hop)
        synchronized(lock) {
            for (i in start until window.size) {
                ring[writePos] = window[i]
                writePos = (writePos + 1) % capacity
                if (filled < capacity) filled++
            }
        }
    }

    /** Number of valid samples currently buffered. */
    fun available(): Int = synchronized(lock) { filled }

    /** Snapshot of the buffered audio in chronological order (oldest → newest). */
    fun snapshot(): FloatArray = synchronized(lock) {
        val out = FloatArray(filled)
        val startPos = if (filled < capacity) 0 else writePos
        for (i in 0 until filled) out[i] = ring[(startPos + i) % capacity]
        out
    }

    fun reset() = synchronized(lock) { writePos = 0; filled = 0 }
}
