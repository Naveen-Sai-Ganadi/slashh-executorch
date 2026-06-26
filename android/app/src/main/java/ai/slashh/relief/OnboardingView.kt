package ai.slashh.relief

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.view.MotionEvent
import android.view.View

/**
 * First-launch onboarding (local profile — no account, no network). The user
 * picks which reliefs they like; the choice is what the router draws from when
 * stress is detected. Defaults to all selected.
 */
class OnboardingView(
    context: Context,
    private val onDone: (Set<ReliefType>) -> Unit,
) : View(context) {

    private val selected = ReliefType.entries.toMutableSet()
    private val types = ReliefType.entries.toList()
    private val chipRects = Array(types.size) { RectF() }
    private val startRect = RectF()

    private var bg: LinearGradient? = null
    private val bgPaint = Paint()
    private val title = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val sub = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF8A93A0.toInt(); textAlign = Paint.Align.CENTER
    }
    private val prompt = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFC7CED8.toInt(); textAlign = Paint.Align.LEFT; isFakeBoldText = true
    }
    private val chipOff = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF161B24.toInt() }
    private val chipOn = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF1E3A34.toInt() }
    private val chipText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.LEFT
    }
    private val check = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.RIGHT
    }
    private val startBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF2E7D5B.toInt() }
    private val startText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val note = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6E7480.toInt(); textAlign = Paint.Align.CENTER
    }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) {
        bg = LinearGradient(0f, 0f, 0f, h.toFloat(), 0xFF14161D.toInt(), 0xFF05060A.toInt(), Shader.TileMode.CLAMP)
        bgPaint.shader = bg
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.drawRect(0f, 0f, w, h, bgPaint)

        title.textSize = w * 0.09f
        canvas.drawText("slashh", w / 2f, h * 0.11f, title)
        sub.textSize = w * 0.04f
        canvas.drawText("private, on-device stress relief", w / 2f, h * 0.155f, sub)

        prompt.textSize = w * 0.048f
        val pad = w * 0.1f
        canvas.drawText("What helps you unwind?", pad, h * 0.24f, prompt)

        val chipH = h * 0.085f
        val gap = h * 0.018f
        var y = h * 0.27f
        chipText.textSize = w * 0.05f
        check.textSize = w * 0.05f
        for (i in types.indices) {
            chipRects[i].set(pad, y, w - pad, y + chipH)
            val on = types[i] in selected
            canvas.drawRoundRect(chipRects[i], chipH * 0.3f, chipH * 0.3f, if (on) chipOn else chipOff)
            val fm = chipText.fontMetrics
            val ty = y + chipH / 2f - (fm.ascent + fm.descent) / 2f
            chipText.color = if (on) 0xFFEAEFF5.toInt() else 0xFF9AA3AF.toInt()
            canvas.drawText("${types[i].emoji}  ${types[i].title}", pad + w * 0.05f, ty, chipText)
            check.color = if (on) 0xFF7FD1A6.toInt() else 0xFF3A4250.toInt()
            canvas.drawText(if (on) "✓" else "+", w - pad - w * 0.05f, ty, check)
            y += chipH + gap
        }

        val bw = w * 0.8f
        val bh = h * 0.075f
        val bx = (w - bw) / 2f
        val by = h * 0.88f
        startRect.set(bx, by, bx + bw, by + bh)
        // dim the button if nothing selected
        startBg.alpha = if (selected.isEmpty()) 90 else 255
        canvas.drawRoundRect(startRect, bh / 2f, bh / 2f, startBg)
        startText.textSize = w * 0.05f
        val fm = startText.fontMetrics
        canvas.drawText("Get started", w / 2f, by + bh / 2f - (fm.ascent + fm.descent) / 2f, startText)

        note.textSize = w * 0.034f
        canvas.drawText("🔒  your choices stay on this device", w / 2f, h * 0.965f, note)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP) return true
        for (i in types.indices) if (chipRects[i].contains(event.x, event.y)) {
            if (types[i] in selected) selected.remove(types[i]) else selected.add(types[i])
            invalidate()
            return true
        }
        if (startRect.contains(event.x, event.y) && selected.isNotEmpty()) {
            onDone(selected.toSet())
        }
        return true
    }
}
