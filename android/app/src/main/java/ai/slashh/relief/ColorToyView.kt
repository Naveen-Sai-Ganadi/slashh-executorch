package ai.slashh.relief

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.os.SystemClock
import android.view.MotionEvent
import kotlin.math.min

/**
 * A calming color-tap toy: each tap blooms a soft expanding ring of a soothing
 * color that fades out. No goal, no score — just gentle sensory play.
 */
class ColorToyView(context: Context, onDismiss: () -> Unit) :
    ReliefOverlayView(context, onDismiss) {

    init { title = "🎨  Color tap" }
    override val animated = true

    private class Ripple(val x: Float, val y: Float, val start: Long, val color: Int)

    private val ripples = ArrayList<Ripple>()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val hint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6E7480.toInt(); textAlign = Paint.Align.CENTER
    }
    private val palette = intArrayOf(
        0xFF4FC3F7.toInt(), 0xFF81C784.toInt(), 0xFFBA68C8.toInt(),
        0xFFFFB74D.toInt(), 0xFF4DD0E1.toInt(), 0xFFF06292.toInt(),
    )
    private val lifeMs = 1800f

    override fun onShow() { ripples.clear() }

    override fun drawContent(canvas: Canvas, w: Float, h: Float) {
        val now = SystemClock.uptimeMillis()
        val maxR = min(w, h) * 0.42f
        val it = ripples.iterator()
        while (it.hasNext()) {
            val rp = it.next()
            val p = (now - rp.start) / lifeMs
            if (p >= 1f) { it.remove(); continue }
            val eased = 1f - (1f - p) * (1f - p)
            paint.color = rp.color
            paint.alpha = (200 * (1f - p)).toInt().coerceIn(0, 255)
            paint.strokeWidth = (min(w, h) * 0.02f) * (1f - p * 0.5f)
            canvas.drawCircle(rp.x, rp.y, maxR * eased, paint)
        }
        if (ripples.isEmpty()) {
            hint.textSize = w * 0.045f
            canvas.drawText("tap anywhere", w / 2f, h * 0.5f, hint)
        }
    }

    override fun onContentTouch(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN || event.action == MotionEvent.ACTION_MOVE) {
            val y = event.y
            if (y > contentTop(height.toFloat()) && y < contentBottom(height.toFloat())) {
                val c = palette[(event.x.toInt() + event.y.toInt()) % palette.size]
                ripples.add(Ripple(event.x, y, SystemClock.uptimeMillis(), c))
                invalidate()
            }
        }
        return true
    }
}
