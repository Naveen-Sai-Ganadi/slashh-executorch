package ai.slashh.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.concurrent.thread

/**
 * Microphone capture → rolling windows. Records 16 kHz mono PCM and emits a
 * fresh [AudioConfig.WINDOW_SAMPLES]-sample window every [AudioConfig.HOP_SAMPLES]
 * samples (3 s window, 1 s hop — matches the training contract).
 *
 * Implemented as a simple ring buffer: read fixed chunks off [AudioRecord], copy
 * them in, and once a full hop of new audio has arrived, snapshot the most recent
 * window and hand it to [onWindow]. The callback runs on the capture thread —
 * keep it light or dispatch heavy work elsewhere.
 *
 * Requires RECORD_AUDIO permission (request before calling [start]).
 */
class AudioCapture(
    private val onWindow: (FloatArray) -> Unit,
) {
    private val sampleRate = AudioConfig.SAMPLE_RATE
    private val windowSamples = AudioConfig.WINDOW_SAMPLES
    private val hopSamples = AudioConfig.HOP_SAMPLES

    private val ring = FloatArray(windowSamples)
    private var writePos = 0          // next write index (mod windowSamples)
    private var filled = 0            // total samples ever written
    private var sinceLastEmit = 0

    @Volatile private var running = false
    private var recorder: AudioRecord? = null
    private var worker: Thread? = null

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufBytes = maxOf(minBuf, hopSamples * 2)   // at least one hop
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufBytes,
        )
        check(rec.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord init failed" }
        recorder = rec
        running = true
        rec.startRecording()

        worker = thread(name = "slashh-audio", isDaemon = true) {
            val chunk = ShortArray(hopSamples)
            while (running) {
                val n = rec.read(chunk, 0, chunk.size)
                if (n <= 0) continue
                for (i in 0 until n) {
                    ring[writePos] = chunk[i] / 32768.0f      // PCM16 → [-1,1)
                    writePos = (writePos + 1) % windowSamples
                    filled++
                    sinceLastEmit++
                }
                if (filled >= windowSamples && sinceLastEmit >= hopSamples) {
                    sinceLastEmit = 0
                    onWindow(snapshot())
                }
            }
        }
    }

    fun stop() {
        running = false
        worker?.join(500)
        worker = null
        recorder?.run { stop(); release() }
        recorder = null
    }

    /** Most-recent window in chronological order (oldest → newest). */
    private fun snapshot(): FloatArray {
        val out = FloatArray(windowSamples)
        // ring[writePos] is the oldest sample once the buffer is full
        for (i in 0 until windowSamples) {
            out[i] = ring[(writePos + i) % windowSamples]
        }
        return out
    }
}
