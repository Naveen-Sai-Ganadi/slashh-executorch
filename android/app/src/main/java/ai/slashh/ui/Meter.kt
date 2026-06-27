package ai.slashh.ui

import ai.slashh.audio.AudioConfig
import ai.slashh.audio.StressPipeline
import kotlin.math.roundToInt

/**
 * Pure presentation logic for the stress meter (M8). Maps a
 * [StressPipeline.StressState] to everything the view needs to draw — and
 * nothing it doesn't — so the mapping (bands, colors, percent, hysteresis
 * agreement) is unit-testable on a plain JVM, with no Android dependency.
 *
 * The view ([StressMeterView]) is then a dumb renderer of [MeterModel].
 */
enum class StressBand { IDLE, CALM, ELEVATED, HIGH }

data class MeterModel(
    /** false before the first voiced window — nothing to show yet */
    val hasReading: Boolean,
    /** 0..100 smoothed stress level */
    val percent: Int,
    val band: StressBand,
    val label: String,
    /** ARGB color for the bar/accent (plain Int — no android.graphics here) */
    val argb: Int,
    /** hysteresis latch, surfaced for the calming cue (M9) */
    val stressed: Boolean,
)

object Meter {
    // Color ramp as ARGB ints so this stays android-free and testable.
    const val COLOR_IDLE = 0xFF9E9E9E.toInt()     // grey
    const val COLOR_CALM = 0xFF2E7D32.toInt()     // green
    const val COLOR_ELEVATED = 0xFFF9A825.toInt() // amber
    const val COLOR_HIGH = 0xFFC62828.toInt()     // red

    /** The mic-permission-denied state — nothing to read, prompt the user. */
    fun needMic(): MeterModel = MeterModel(
        hasReading = false,
        percent = 0,
        band = StressBand.IDLE,
        label = "Mic needed",
        argb = COLOR_IDLE,
        stressed = false,
    )

    /** Shown while the background monitor owns the mic (in-app meter is paused). */
    fun monitoring(): MeterModel = MeterModel(
        hasReading = false,
        percent = 0,
        band = StressBand.IDLE,
        label = "Monitoring in background",
        argb = COLOR_CALM,
        stressed = false,
    )

    /** Build the display model for one pipeline state. */
    fun from(state: StressPipeline.StressState): MeterModel {
        val level = state.level
            ?: return MeterModel(
                hasReading = false,
                percent = 0,
                band = StressBand.IDLE,
                label = "Listening…",
                argb = COLOR_IDLE,
                stressed = false,
            )

        val percent = (level * 100f).roundToInt().coerceIn(0, 100)

        // Band agrees with the pipeline's hysteresis latch so the UI can never
        // contradict the decision loop: a latched-stressed state always reads
        // HIGH, even while the level sits in the hysteresis band on its way down.
        val band = when {
            state.stressed -> StressBand.HIGH
            level >= AudioConfig.RELEASE_THRESHOLD -> StressBand.ELEVATED
            else -> StressBand.CALM
        }

        val (label, argb) = when (band) {
            StressBand.HIGH -> "High stress" to COLOR_HIGH
            StressBand.ELEVATED -> "Elevated" to COLOR_ELEVATED
            StressBand.CALM -> "Calm" to COLOR_CALM
            StressBand.IDLE -> "Listening…" to COLOR_IDLE
        }

        return MeterModel(
            hasReading = true,
            percent = percent,
            band = band,
            label = label,
            argb = argb,
            stressed = state.stressed,
        )
    }
}
