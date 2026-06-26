package ai.slashh.audio

/**
 * Maps a flattened `[1,1,N_MELS,N_FRAMES]` log-mel feature buffer to a single
 * stress probability in `[0,1]`. Kept as an interface so the audio pipeline (M6)
 * is testable with a stub, while the real ExecuTorch runtime (M7) plugs in
 * behind it. See [ExecuTorchStressClassifier].
 */
fun interface StressClassifier {
    /** @param features row-major log-mel, length N_MELS*N_FRAMES. @return score in [0,1]. */
    fun score(features: FloatArray): Float
}
