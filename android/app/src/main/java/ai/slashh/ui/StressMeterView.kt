package ai.slashh.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * A dumb renderer for [MeterModel] (M8): a rounded track, a band-colored fill
 * proportional to `percent`, a big percentage read-out, and the band label.
 * All decisions live in [Meter]; this class only draws. Call [render] from the
 * UI thread with each new model.
 */
class StressMeterView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private var model: MeterModel = Meter.from(
        ai.slashh.audio.StressPipeline.StressState(false, null, null, false)
    )

    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF26282B.toInt() }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val pctText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val labelText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFB0B3B8.toInt(); textAlign = Paint.Align.CENTER
    }
    private val rect = RectF()

    fun render(m: MeterModel) {
        model = m
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        setBackgroundColor(0xFF161719.toInt())

        // big percentage
        pctText.textSize = min(w, h) * 0.26f
        val pctStr = if (model.hasReading) "${model.percent}%" else "—"
        canvas.drawText(pctStr, w / 2f, h * 0.42f, pctText)

        // label
        labelText.textSize = min(w, h) * 0.07f
        labelText.color = if (model.hasReading) model.argb else 0xFFB0B3B8.toInt()
        canvas.drawText(model.label, w / 2f, h * 0.55f, labelText)

        // track + proportional fill bar
        val barH = h * 0.10f
        val top = h * 0.68f
        val pad = w * 0.10f
        val radius = barH / 2f
        rect.set(pad, top, w - pad, top + barH)
        canvas.drawRoundRect(rect, radius, radius, track)

        val frac = (model.percent / 100f).coerceIn(0f, 1f)
        if (model.hasReading && frac > 0f) {
            fill.color = model.argb
            rect.set(pad, top, pad + (w - 2 * pad) * frac, top + barH)
            canvas.drawRoundRect(rect, radius, radius, fill)
        }
    }
}
