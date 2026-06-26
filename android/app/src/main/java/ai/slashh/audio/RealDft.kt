package ai.slashh.audio

import kotlin.math.cos
import kotlin.math.sin

/**
 * Exact real-input DFT producing `n/2 + 1` complex bins — the on-device twin of
 * NumPy's `rfft(frame, n=N_FFT)` that torchaudio uses inside `MelSpectrogram`.
 *
 * `N_FFT = 400` is not a power of two, so we can't use a radix-2 FFT and we must
 * NOT zero-pad to 512 (that would shift the bin frequencies and break parity).
 * A direct DFT with precomputed twiddle tables is exact to float precision and,
 * at ~48M mul-adds per 3 s window computed once per 1 s hop, is well within the
 * real-time budget on a Snapdragon-class CPU. Swap for a mixed-radix FFT only if
 * profiling later says the feature step is a bottleneck.
 */
class RealDft(private val n: Int) {
    private val bins = n / 2 + 1
    // cosTable[k][t] = cos(2π k t / n), sinTable likewise. ~201*400 doubles.
    private val cosTable = Array(bins) { DoubleArray(n) }
    private val sinTable = Array(bins) { DoubleArray(n) }

    init {
        val twoPiOverN = 2.0 * Math.PI / n
        for (k in 0 until bins) {
            val c = cosTable[k]
            val s = sinTable[k]
            for (t in 0 until n) {
                val ang = twoPiOverN * k * t
                c[t] = cos(ang)
                s[t] = sin(ang)
            }
        }
    }

    /** Power spectrum |X_k|^2 (matches torchaudio `power=2.0`). `frame.size == n`. */
    fun powerSpectrum(frame: DoubleArray, out: DoubleArray) {
        require(frame.size == n) { "frame size ${frame.size} != $n" }
        for (k in 0 until bins) {
            val c = cosTable[k]
            val s = sinTable[k]
            var re = 0.0
            var im = 0.0
            for (t in 0 until n) {
                val x = frame[t]
                re += x * c[t]
                im -= x * s[t]      // e^{-j…}: imaginary part is -sin
            }
            out[k] = re * re + im * im
        }
    }

    fun binCount(): Int = bins
}
