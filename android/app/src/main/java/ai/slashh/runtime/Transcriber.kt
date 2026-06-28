package ai.slashh.runtime

/**
 * Turns a mono PCM float window into text — the on-device ASR seam. The monitor feeds an
 * audio buffer to [transcribe]; the resulting transcript drives the text-stress classifier,
 * whose score is fused with the audio model's score for a more confident reading.
 *
 * Implementations are **honest**: when the model/helper isn't available they return `""`
 * rather than fabricating a transcript. The real implementation ([NpuWhisperTranscriber])
 * runs Whisper on the Hexagon NPU out-of-process; there is no in-app fake.
 *
 * Ported from the ScamShield/Edge `Transcriber` interface; the old `ExecuTorchModule.Backend`
 * dependency was replaced with the small local [Backend] enum below (the Edge ExecuTorchModule
 * was an unused TODO stub).
 */
interface Transcriber {
    /** @param samples mono PCM float in [-1, 1] at [WhisperConfig.SAMPLE_RATE]. */
    suspend fun transcribe(samples: FloatArray): String

    /** Which backend serviced the last call (for the honest stats panel / UI badge). */
    val backend: Backend

    /** Compute backend that actually serviced the last transcription. */
    enum class Backend { QNN_NPU, FAKE }
}
