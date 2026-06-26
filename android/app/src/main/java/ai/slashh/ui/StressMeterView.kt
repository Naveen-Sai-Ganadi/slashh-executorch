package ai.slashh.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sin

/**
 * Premium renderer for [MeterModel] (M8): a circular arc gauge with a band-color
 * glow, an animated percentage read-out, a custom header, and an on-device
 * privacy badge. All *decisions* still live in [Meter]; this class only draws.
 *
 * The gauge value eases toward the target each frame, and the idle ("Listening…")
 * state shows a soft pulse — both driven off [SystemClock.uptimeMillis] via
 * postInvalidateOnAnimation, so there's no timer to manage.
 */
class StressMeterView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private var model: MeterModel = Meter.from(
        ai.slashh.audio.StressPipeline.StressState(false, null, null, false)
    )

    // Eased value the gauge actually draws (lerps toward model.percent).
    private var animPercent = 0f

    private val START_ANGLE = 135f
    private val SWEEP_FULL = 270f
    private val BG_TOP = 0xFF14161D.toInt()
    private val BG_BOTTOM = 0xFF05060A.toInt()
    private val TRACK = 0xFF20242C.toInt()

    private var bgShader: LinearGradient? = null
    private val bgPaint = Paint()
    private val trackArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; color = TRACK
    }
    private val glowArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val progArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val number = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val unit = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF8A8F98.toInt(); textAlign = Paint.Align.CENTER
    }
    private val bandLabel = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val wordmark = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val subtitle = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6E7480.toInt(); textAlign = Paint.Align.CENTER
    }
    private val badgeBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF161A22.toInt() }
    private val badgeText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF7FD1A6.toInt(); textAlign = Paint.Align.CENTER
    }
    private val oval = RectF()
    private val badgeRect = RectF()

    fun render(m: MeterModel) {
        model = m
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) {
        bgShader = LinearGradient(0f, 0f, 0f, h.toFloat(), BG_TOP, BG_BOTTOM, Shader.TileMode.CLAMP)
        bgPaint.shader = bgShader
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        canvas.drawRect(0f, 0f, w, h, bgPaint)

        // ---- header ---------------------------------------------------------
        wordmark.textSize = w * 0.085f
        canvas.drawText("slashh", w / 2f, h * 0.11f, wordmark)
        subtitle.textSize = w * 0.036f
        canvas.drawText("on-device voice check-in", w / 2f, h * 0.155f, subtitle)

        // ---- gauge geometry -------------------------------------------------
        val cx = w / 2f
        val cy = h * 0.44f
        val radius = min(w * 0.36f, h * 0.22f)
        val stroke = radius * 0.16f
        oval.set(cx - radius, cy - radius, cx + radius, cy + radius)
        trackArc.strokeWidth = stroke
        glowArc.strokeWidth = stroke * 1.7f
        progArc.strokeWidth = stroke

        // ease the displayed value toward the target
        val target = if (model.hasReading) model.percent.toFloat() else 0f
        animPercent += (target - animPercent) * 0.18f
        if (abs(target - animPercent) < 0.3f) animPercent = target

        // track
        canvas.drawArc(oval, START_ANGLE, SWEEP_FULL, false, trackArc)

        val accent = if (model.hasReading) model.argb else Meter.COLOR_IDLE

        if (model.hasReading) {
            val sweep = SWEEP_FULL * (animPercent / 100f).coerceIn(0f, 1f)
            glowArc.color = (accent and 0x00FFFFFF) or 0x55000000      // translucent glow
            canvas.drawArc(oval, START_ANGLE, sweep, false, glowArc)
            progArc.color = accent
            canvas.drawArc(oval, START_ANGLE, sweep, false, progArc)

            // big number
            number.textSize = radius * 0.78f
            canvas.drawText("${animPercent.toInt()}", cx, cy + radius * 0.12f, number)
            unit.textSize = radius * 0.22f
            canvas.drawText("%", cx, cy + radius * 0.45f, unit)
            // band label
            bandLabel.textSize = radius * 0.24f
            bandLabel.color = accent
            canvas.drawText(model.label.uppercase(), cx, cy + radius * 1.55f, bandLabel)
        } else {
            // idle: soft pulsing accent arc + listening label with animated dots
            val phase = (SystemClock.uptimeMillis() % 1600L) / 1600f
            val pulse = 0.5f - 0.5f * kotlin.math.cos(phase * 2f * Math.PI.toFloat())
            glowArc.color = (Meter.COLOR_IDLE and 0x00FFFFFF) or ((40 + 60 * pulse).toInt() shl 24)
            canvas.drawArc(oval, START_ANGLE, SWEEP_FULL, false, glowArc)
            number.textSize = radius * 0.55f
            number.alpha = (120 + 100 * pulse).toInt().coerceIn(0, 255)
            canvas.drawText("—", cx, cy + radius * 0.20f, number)
            number.alpha = 255
            val dots = ".".repeat(1 + ((SystemClock.uptimeMillis() / 400L) % 3L).toInt())
            bandLabel.textSize = radius * 0.24f
            bandLabel.color = 0xFF8A8F98.toInt()
            canvas.drawText("Listening$dots", cx, cy + radius * 1.55f, bandLabel)
        }

        // ---- on-device privacy badge ---------------------------------------
        badgeText.textSize = w * 0.036f
        val bt = "🔒  100% on-device · no network"
        val tw = badgeText.measureText(bt)
        val bw = tw + w * 0.10f
        val bh = h * 0.045f
        val bx = (w - bw) / 2f
        val by = h * 0.88f
        badgeRect.set(bx, by, bx + bw, by + bh)
        canvas.drawRoundRect(badgeRect, bh / 2f, bh / 2f, badgeBg)
        val fm = badgeText.fontMetrics
        canvas.drawText(bt, w / 2f, by + bh / 2f - (fm.ascent + fm.descent) / 2f, badgeText)

        // keep animating while easing or idle-pulsing
        if (!model.hasReading || animPercent != target) postInvalidateOnAnimation()
    }
}
