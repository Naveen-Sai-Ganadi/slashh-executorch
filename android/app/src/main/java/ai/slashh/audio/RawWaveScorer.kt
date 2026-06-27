package ai.slashh.audio

/**
 * Maps a raw 16 kHz pcm window (length [AudioConfig.WINDOW_SAMPLES]) straight to a
 * stress probability in `[0,1]`. This is the alternative front-end to
 * [StressClassifier]: the tiny StressNet consumes precomputed log-mel features,
 * but the large self-attention teachers (WavLM, …) take the raw waveform and do
 * normalization + feature extraction *inside* the graph. When a [StressPipeline]
 * is given a [RawWaveScorer] it skips the host-side log-mel step entirely.
 *
 * See [WavLmStressClassifier].
 */
fun interface RawWaveScorer {
    /** @param pcm raw window, length [AudioConfig.WINDOW_SAMPLES]. @return score in [0,1]. */
    fun scoreWindow(pcm: FloatArray): Float
}
