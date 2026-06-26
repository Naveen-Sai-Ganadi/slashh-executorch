package ai.slashh.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

/**
 * On-device log-mel extractor — the native twin of `model/features.py`
 * (torchaudio `MelSpectrogram` + `log10`). Verified against torchaudio to
 * < 1e-5 max-abs-error by the reference algorithm and parity-tested on the JVM
 * against golden vectors (`LogMelParityTest`, M6 acceptance).
 *
 * Pipeline (all parameters from [AudioConfig]):
 *   reflect-pad by N_FFT/2  →  framed (hop=HOP_LENGTH)  →  periodic Hann
 *   →  power spectrum (RealDft)  →  HTK mel filterbank  →  log10(mel + eps)
 *
 * Output is the fixed `[N_MELS][N_FRAMES]` feature map; flatten row-major into
 * the `[1,1,N_MELS,N_FRAMES]` tensor the `.pte` expects.
 */
class LogMel {
    private val nFft = AudioConfig.N_FFT
    private val hop = AudioConfig.HOP_LENGTH
    private val nMels = AudioConfig.N_MELS
    private val nFrames = AudioConfig.N_FRAMES
    private val pad = nFft / 2

    private val dft = RealDft(nFft)
    private val nBins = dft.binCount()
    private val window: DoubleArray = hannPeriodic(AudioConfig.WIN_LENGTH)
    // melFb[bin][mel] — applied as melFb^T · powerSpectrum.
    private val melFb: Array<DoubleArray> = melFilterbank(
        nBins, AudioConfig.F_MIN, AudioConfig.F_MAX, nMels, AudioConfig.SAMPLE_RATE
    )

    // scratch buffers reused per call (single-threaded use)
    private val frame = DoubleArray(nFft)
    private val power = DoubleArray(nBins)

    /**
     * Extract from a single window. `pcm` should be [AudioConfig.WINDOW_SAMPLES]
     * float samples; shorter/longer inputs are padded/truncated to keep the frame
     * count fixed, exactly as `features.extract` does.
     *
     * Returns `[N_MELS][N_FRAMES]`.
     */
    fun extract(pcm: FloatArray): Array<FloatArray> {
        val padded = reflectPad(pcm, pad)
        val out = Array(nMels) { FloatArray(nFrames) }

        for (i in 0 until nFrames) {
            val start = i * hop
            // windowed frame (zero-fill if we ran past the padded signal)
            for (t in 0 until nFft) {
                val idx = start + t
                val x = if (idx < padded.size) padded[idx] else 0.0
                frame[t] = x * window[t]
            }
            dft.powerSpectrum(frame, power)
            // mel projection: out[m, i] = log10( sum_b melFb[b][m]*power[b] + eps )
            for (m in 0 until nMels) {
                var acc = 0.0
                for (b in 0 until nBins) {
                    acc += melFb[b][m] * power[b]
                }
                out[m][i] = log10(acc + AudioConfig.LOG_EPS).toFloat()
            }
        }
        return out
    }

    /** Row-major flatten into the model input layout `[1,1,N_MELS,N_FRAMES]`. */
    fun extractFlat(pcm: FloatArray): FloatArray {
        val mat = extract(pcm)
        val flat = FloatArray(nMels * nFrames)
        var k = 0
        for (m in 0 until nMels) {
            val row = mat[m]
            for (i in 0 until nFrames) flat[k++] = row[i]
        }
        return flat
    }

    private companion object {
        /** Periodic Hann window (torch.hann_window default, periodic=True). */
        fun hannPeriodic(n: Int): DoubleArray =
            DoubleArray(n) { 0.5 - 0.5 * cos(2.0 * PI * it / n) }

        /** Reflect padding matching numpy/torch `pad_mode="reflect"` (no edge). */
        fun reflectPad(x: FloatArray, pad: Int): DoubleArray {
            val n = x.size
            val out = DoubleArray(n + 2 * pad)
            for (i in 0 until n) out[pad + i] = x[i].toDouble()
            for (p in 1..pad) {
                out[pad - p] = x[p].toDouble()                 // left reflect
                out[pad + n - 1 + p] = x[n - 1 - p].toDouble() // right reflect
            }
            return out
        }

        fun hzToMelHtk(f: Double): Double = 2595.0 * log10(1.0 + f / 700.0)
        fun melToHzHtk(m: Double): Double = 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0)

        /**
         * HTK triangular mel filterbank, `norm=None` — matches
         * `torchaudio.functional.melscale_fbanks`. Returns `[nBins][nMels]`.
         */
        fun melFilterbank(
            nBins: Int, fMin: Double, fMax: Double, nMels: Int, sr: Int
        ): Array<DoubleArray> {
            // bin center frequencies, linearly spaced 0 .. sr/2
            val allFreqs = DoubleArray(nBins) { it * (sr / 2.0) / (nBins - 1) }
            val mMin = hzToMelHtk(fMin)
            val mMax = hzToMelHtk(fMax)
            val fPts = DoubleArray(nMels + 2) {
                melToHzHtk(mMin + (mMax - mMin) * it / (nMels + 1))
            }
            val fDiff = DoubleArray(nMels + 1) { fPts[it + 1] - fPts[it] }

            val fb = Array(nBins) { DoubleArray(nMels) }
            for (b in 0 until nBins) {
                val freq = allFreqs[b]
                for (m in 0 until nMels) {
                    val down = -(fPts[m] - freq) / fDiff[m]          // up-slope of triangle m
                    val up = (fPts[m + 2] - freq) / fDiff[m + 1]     // down-slope
                    fb[b][m] = max(0.0, min(down, up))
                }
            }
            return fb
        }
    }
}
