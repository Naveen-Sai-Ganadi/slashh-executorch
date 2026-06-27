package ai.slashh.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View

/**
 * On-the-fly, on-device threshold calibration. Two steps — speak calmly, then
 * speak stressed — and [Calibration] sets a personal enter/exit threshold for
 * this voice + device. The audio pipeline keeps running behind this overlay;
 * MainActivity forwards each voiced model score via [feedScore].
 */
class CalibrationView(
    context: Context,
    private val onDone: (calmAnchor: Float, stressAnchor: Float) -> Unit,
    private val onCancel: () -> Unit,
) : View(context) {

    private enum class Phase { INTRO, CALM, STRESSED, RESULT }
    private var phase = Phase.INTRO
    private val calm = ArrayList<Float>()
    private val stressed = ArrayList<Float>()
    private var result: Calibration.Result? = null
    private val target = 6     // ~6 voiced windows (~6 s) per step

    init { isClickable = true }

    /** True while we should be collecting mic scores. */
    fun isCollecting() = phase == Phase.CALM || phase == Phase.STRESSED

    /** MainActivity calls this with each voiced model score. */
    fun feedScore(raw: Float) {
        when (phase) {
            Phase.CALM -> { calm.add(raw); if (calm.size >= target) phase = Phase.STRESSED }
            Phase.STRESSED -> {
                stressed.add(raw)
                if (stressed.size >= target) {
                    result = Calibration.compute(calm, stressed)
                    phase = Phase.RESULT
                }
            }
            else -> {}
        }
        invalidate()
    }

    private val scrim = Paint().apply { color = 0xFF0A0E14.toInt() }
    private val title = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val big = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val sub = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF9AA3AF.toInt(); textAlign = Paint.Align.CENTER
    }
    private val dotOn = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF7FD1A6.toInt() }
    private val dotOff = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF263042.toInt() }
    private val btnBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF2E7D5B.toInt() }
    private val btn2Bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF18202C.toInt() }
    private val btnText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val cancel = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6E7480.toInt(); textAlign = Paint.Align.RIGHT
    }
    private val primaryRect = RectF()
    private val secondaryRect = RectF()
    private val cancelRect = RectF()

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        canvas.drawRect(0f, 0f, w, h, scrim)

        cancel.textSize = w * 0.045f
        canvas.drawText("✕", w * 0.92f, h * 0.08f, cancel)
        cancelRect.set(w * 0.82f, h * 0.04f, w, h * 0.11f)

        title.textSize = w * 0.058f
        big.textSize = w * 0.085f
        sub.textSize = w * 0.044f

        when (phase) {
            Phase.INTRO -> {
                canvas.drawText("🎚  Calibrate to your voice", w / 2f, h * 0.30f, title)
                sub.textSize = w * 0.046f
                canvas.drawText("Two quick steps so Slashh learns", w / 2f, h * 0.40f, sub)
                canvas.drawText("your normal vs stressed voice.", w / 2f, h * 0.445f, sub)
                primaryButton(canvas, w, h, "Start")
            }
            Phase.CALM -> step(canvas, w, h, "Step 1 of 2", "Speak normally  🗣",
                "talk calmly for a few seconds", calm.size)
            Phase.STRESSED -> step(canvas, w, h, "Step 2 of 2", "Now sound stressed  😣",
                "raise your voice — faster, tenser", stressed.size)
            Phase.RESULT -> result(canvas, w, h)
        }
    }

    private fun step(canvas: Canvas, w: Float, h: Float, head: String, prompt: String,
                     hint: String, count: Int) {
        canvas.drawText(head, w / 2f, h * 0.20f, title)
        canvas.drawText(prompt, w / 2f, h * 0.46f, big)
        canvas.drawText(hint, w / 2f, h * 0.53f, sub)
        // progress dots
        val n = target
        val r = w * 0.012f
        val gap = w * 0.05f
        val startX = w / 2f - (n - 1) * gap / 2f
        for (i in 0 until n) {
            canvas.drawCircle(startX + i * gap, h * 0.66f, r, if (i < count) dotOn else dotOff)
        }
        sub.textSize = w * 0.038f
        canvas.drawText("listening…", w / 2f, h * 0.73f, sub)
    }

    private fun result(canvas: Canvas, w: Float, h: Float) {
        val res = result ?: return
        val ok = res.separable
        canvas.drawText(if (ok) "Calibrated ✓" else "Almost — try again", w / 2f, h * 0.22f, title)

        sub.textSize = w * 0.05f
        canvas.drawText("calm reads   %.2f".format(res.calmAnchor), w / 2f, h * 0.36f, sub)
        canvas.drawText("stressed reads   %.2f".format(res.stressAnchor), w / 2f, h * 0.42f, sub)
        sub.color = 0xFF7FD1A6.toInt()
        canvas.drawText("tuned to your voice ✓", w / 2f, h * 0.49f, sub)
        sub.color = 0xFF9AA3AF.toInt()
        if (!ok) {
            sub.textSize = w * 0.042f
            canvas.drawText("Your two voices were too similar.", w / 2f, h * 0.57f, sub)
            canvas.drawText("Redo with a bigger difference.", w / 2f, h * 0.615f, sub)
        }
        primaryButton(canvas, w, h, if (ok) "Use this" else "Use anyway")
        // secondary: redo
        val bw = w * 0.5f; val bh = h * 0.06f; val bx = (w - bw) / 2f; val by = h * 0.80f
        secondaryRect.set(bx, by, bx + bw, by + bh)
        canvas.drawRoundRect(secondaryRect, bh / 2f, bh / 2f, btn2Bg)
        btnText.textSize = w * 0.044f; btnText.color = 0xFFB7C0CC.toInt()
        val fm = btnText.fontMetrics
        canvas.drawText("Redo", w / 2f, by + bh / 2f - (fm.ascent + fm.descent) / 2f, btnText)
        btnText.color = Color.WHITE
    }

    private fun primaryButton(canvas: Canvas, w: Float, h: Float, label: String) {
        val bw = w * 0.7f; val bh = h * 0.072f; val bx = (w - bw) / 2f; val by = h * 0.70f
        primaryRect.set(bx, by, bx + bw, by + bh)
        canvas.drawRoundRect(primaryRect, bh / 2f, bh / 2f, btnBg)
        btnText.textSize = w * 0.05f
        val fm = btnText.fontMetrics
        canvas.drawText(label, w / 2f, by + bh / 2f - (fm.ascent + fm.descent) / 2f, btnText)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP) return true
        val x = event.x; val y = event.y
        if (cancelRect.contains(x, y)) { onCancel(); return true }
        when (phase) {
            Phase.INTRO -> if (primaryRect.contains(x, y)) {
                calm.clear(); stressed.clear(); phase = Phase.CALM; invalidate()
            }
            Phase.RESULT -> {
                if (primaryRect.contains(x, y)) result?.let { onDone(it.calmAnchor, it.stressAnchor) }
                else if (secondaryRect.contains(x, y)) {
                    calm.clear(); stressed.clear(); result = null; phase = Phase.CALM; invalidate()
                }
            }
            else -> {}
        }
        return true
    }
}
