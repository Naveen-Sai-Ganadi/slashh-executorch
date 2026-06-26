package ai.slashh.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import kotlin.math.min

/**
 * The calming intervention overlay (M9): a dimmed full-screen view with a
 * slowly breathing circle and a "Breathe in / out" cue. Tap anywhere to
 * dismiss. *When* it appears is decided by [CalmCue]; this view only animates
 * and reports dismissal.
 *
 * A 4s-in / 4s-out cycle (box-ish breathing) paces the circle between a small
 * and large radius. Driven off [SystemClock.uptimeMillis] so it is independent
 * of wall-clock changes.
 */
class BreathOverlayView(
    context: Context,
    private val onDismiss: () -> Unit,
) : View(context) {

    private val inhaleMs = 4_000f
    private val exhaleMs = 4_000f
    private val cycleMs = inhaleMs + exhaleMs

    private val scrim = Paint().apply { color = 0xCC101216.toInt() } // ~80% dim
    private val circle = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF4FC3F7.toInt() }
    private val cue = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER
    }
    private val hint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFB0B3B8.toInt(); textAlign = Paint.Align.CENTER
    }

    private var startMs = 0L

    /** Show the overlay and give a single haptic nudge. */
    fun show() {
        if (visibility == VISIBLE) return
        startMs = SystemClock.uptimeMillis()
        visibility = VISIBLE
        performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
        postInvalidateOnAnimation()
    }

    fun hide() {
        if (visibility != VISIBLE) return
        visibility = GONE
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP) {
            onDismiss()
            return true
        }
        return true // swallow touches so nothing behind reacts
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.drawRect(0f, 0f, w, h, scrim)

        val t = ((SystemClock.uptimeMillis() - startMs) % cycleMs.toLong()).toFloat()
        val inhaling = t < inhaleMs
        // eased 0..1 progress within the current half-cycle
        val p = if (inhaling) t / inhaleMs else 1f - (t - inhaleMs) / exhaleMs
        val eased = 0.5f - 0.5f * kotlin.math.cos(p * Math.PI.toFloat()) // smoothstep-ish

        val rMin = min(w, h) * 0.12f
        val rMax = min(w, h) * 0.34f
        val r = rMin + (rMax - rMin) * eased
        circle.alpha = (90 + 120 * eased).toInt().coerceIn(0, 255)
        canvas.drawCircle(w / 2f, h * 0.45f, r, circle)

        cue.textSize = min(w, h) * 0.08f
        canvas.drawText(if (inhaling) "Breathe in…" else "Breathe out…", w / 2f, h * 0.78f, cue)

        hint.textSize = min(w, h) * 0.045f
        canvas.drawText("tap to dismiss", w / 2f, h * 0.88f, hint)

        if (visibility == VISIBLE) postInvalidateOnAnimation()
    }
}
