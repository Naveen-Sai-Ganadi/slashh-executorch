package ai.slashh.relief

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.view.MotionEvent
import java.util.Random

/**
 * Calming sounds: white / brown noise generated on-device with [AudioTrack]
 * (looped static buffer — no bundled audio, no network). Tap a button to play;
 * the active one is highlighted. Audio is released when the overlay hides.
 */
class SoundsView(context: Context, onDismiss: () -> Unit) :
    ReliefOverlayView(context, onDismiss) {

    init { title = "🎧  Calming sounds" }

    private val buttons = listOf("🌊  White noise", "🐻  Brown noise", "⏸  Stop")
    private val rects = Array(3) { RectF() }
    private var active = -1   // -1 = stopped

    private val btnBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF18202C.toInt() }
    private val btnOn = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF1E3A34.toInt() }
    private val btnText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFEAEFF5.toInt(); textAlign = Paint.Align.CENTER
    }

    private var track: AudioTrack? = null

    override fun onHide() { stopAudio() }

    private fun stopAudio() {
        track?.let { runCatching { it.stop() }; it.release() }
        track = null
        active = -1
    }

    private fun playNoise(brown: Boolean) {
        stopAudio()
        val sr = 44_100
        val n = sr * 2                       // 2-second seamless loop
        val buf = ShortArray(n)
        val rnd = Random()
        var last = 0f
        for (i in 0 until n) {
            val w = rnd.nextFloat() * 2f - 1f
            val s = if (brown) { last = (last + 0.02f * w).coerceIn(-1f, 1f); last * 3.2f }
                    else w * 0.5f
            buf[i] = (s.coerceIn(-1f, 1f) * Short.MAX_VALUE).toInt().toShort()
        }
        @Suppress("DEPRECATION")
        val t = AudioTrack(
            AudioManager.STREAM_MUSIC, sr,
            AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
            n * 2, AudioTrack.MODE_STATIC,
        )
        t.write(buf, 0, n)
        t.setLoopPoints(0, n, -1)
        t.play()
        track = t
    }

    override fun drawContent(canvas: Canvas, w: Float, h: Float) {
        val bw = w * 0.7f
        val bh = h * 0.08f
        val gap = h * 0.03f
        val x = (w - bw) / 2f
        var y = contentTop(h) + h * 0.06f
        btnText.textSize = w * 0.05f
        for (i in 0..2) {
            rects[i].set(x, y, x + bw, y + bh)
            canvas.drawRoundRect(rects[i], bh / 2f, bh / 2f, if (i == active) btnOn else btnBg)
            val fm = btnText.fontMetrics
            btnText.color = if (i == active) 0xFF8FE3B6.toInt() else 0xFFEAEFF5.toInt()
            canvas.drawText(buttons[i], w / 2f, y + bh / 2f - (fm.ascent + fm.descent) / 2f, btnText)
            y += bh + gap
        }
    }

    override fun onContentTouch(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP) return true
        for (i in 0..2) if (rects[i].contains(event.x, event.y)) {
            when (i) {
                0 -> { playNoise(brown = false); active = 0 }
                1 -> { playNoise(brown = true); active = 1 }
                2 -> stopAudio()
            }
            invalidate()
        }
        return true
    }
}
