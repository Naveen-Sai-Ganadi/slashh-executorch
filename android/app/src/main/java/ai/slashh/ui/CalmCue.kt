package ai.slashh.ui

/**
 * Decides *when* to surface the calming breathing prompt (M9) — the trigger and
 * rate-limit logic only, no Android. The view ([BreathOverlayView]) just shows
 * or hides what this returns.
 *
 * Rules:
 * - **Sustained, not twitchy**: stress must hold for [sustainMs] before we
 *   interrupt — a one-window spike shouldn't throw up an overlay.
 * - **Rate-limited**: at most one prompt per [cooldownMs], measured from the
 *   last time one was shown/dismissed, so we don't nag.
 * - **Dismissible**: once dismissed, we stay quiet for the rest of this stress
 *   episode (until stress clears) and respect the cooldown afterwards.
 *
 * The clock is injected (`nowMs`) so this is deterministic and unit-testable on
 * a plain JVM; on device, pass `System.currentTimeMillis()`.
 */
class CalmCue(
    private val sustainMs: Long = 4_000,
    private val cooldownMs: Long = 60_000,
) {
    private var stressedSinceMs: Long? = null
    private var lastShownMs: Long? = null
    private var visible = false
    private var dismissedThisEpisode = false

    /** True the moment the overlay transitions hidden → visible (fire haptic). */
    var justTriggered = false
        private set

    /**
     * Feed the latest hysteresis latch and clock; returns whether the breathing
     * overlay should currently be visible.
     */
    fun onState(stressed: Boolean, nowMs: Long): Boolean {
        justTriggered = false

        if (!stressed) {
            // stress cleared: end the episode, drop the overlay
            stressedSinceMs = null
            dismissedThisEpisode = false
            visible = false
            return false
        }

        if (stressedSinceMs == null) stressedSinceMs = nowMs
        if (visible) return true                 // stay up until dismissed
        if (dismissedThisEpisode) return false   // user said no; wait out the episode

        val sustained = nowMs - (stressedSinceMs ?: nowMs) >= sustainMs
        val cooledDown = lastShownMs?.let { nowMs - it >= cooldownMs } ?: true
        if (sustained && cooledDown) {
            visible = true
            justTriggered = true
            lastShownMs = nowMs
            return true
        }
        return false
    }

    /** User dismissed the overlay; respect cooldown and stay quiet this episode. */
    fun dismiss(nowMs: Long) {
        visible = false
        dismissedThisEpisode = true
        lastShownMs = nowMs
    }

    val isVisible: Boolean get() = visible

    /** Reset all state (e.g. screen backgrounded). */
    fun reset() {
        stressedSinceMs = null
        lastShownMs = null
        visible = false
        dismissedThisEpisode = false
        justTriggered = false
    }
}
