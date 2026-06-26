package ai.slashh.relief

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View

/**
 * Shared chrome for every relief: a calming scrim, a title, and a single
 * "I feel better ✓" pill that dismisses (so it never conflicts with content
 * taps in the games). Subclasses implement [drawContent] / [onContentTouch].
 */
abstract class ReliefOverlayView(
    context: Context,
    private val onDismiss: () -> Unit,
) : View(context), ReliefScreen {

    init { isClickable = true; isFocusable = true }

    private val scrim = Paint().apply { color = 0xFF0A0E14.toInt() } // fully opaque
    private val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val pillBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF18202C.toInt() }
    private val pillText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF8FE3B6.toInt(); textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val pill = RectF()

    /** subclasses set true for ripple/breathing animation loops */
    protected open val animated: Boolean = false
    protected var title: String = ""

    override fun show() {
        if (visibility == VISIBLE) return
        onShow()
        visibility = VISIBLE
        performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
        invalidate()
    }

    override fun hide() {
        if (visibility != VISIBLE) return
        visibility = GONE
        onHide()
    }

    protected open fun onShow() {}
    protected open fun onHide() {}

    protected fun contentTop(h: Float) = h * 0.19f
    protected fun contentBottom(h: Float) = h * 0.80f

    abstract fun drawContent(canvas: Canvas, w: Float, h: Float)
    open fun onContentTouch(event: MotionEvent): Boolean = true

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.drawRect(0f, 0f, w, h, scrim)

        titlePaint.textSize = w * 0.062f
        canvas.drawText(title, w / 2f, h * 0.125f, titlePaint)

        drawContent(canvas, w, h)

        pillText.textSize = w * 0.044f
        val label = "I feel better  ✓"
        val tw = pillText.measureText(label)
        val pw = tw + w * 0.14f
        val ph = h * 0.062f
        val px = (w - pw) / 2f
        val py = h * 0.865f
        pill.set(px, py, px + pw, py + ph)
        canvas.drawRoundRect(pill, ph / 2f, ph / 2f, pillBg)
        val fm = pillText.fontMetrics
        canvas.drawText(label, w / 2f, py + ph / 2f - (fm.ascent + fm.descent) / 2f, pillText)

        if (animated && visibility == VISIBLE) postInvalidateOnAnimation()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP && pill.contains(event.x, event.y)) {
            onDismiss()
            return true
        }
        return onContentTouch(event)
    }
}
