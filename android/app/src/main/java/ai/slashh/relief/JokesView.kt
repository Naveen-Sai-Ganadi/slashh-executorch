package ai.slashh.relief

import android.content.Context
import android.graphics.Canvas
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.view.MotionEvent

/**
 * A quick laugh: shows one (clean, groan-worthy) joke at a time. Tap the card
 * to get another. All bundled locally — no network.
 */
class JokesView(context: Context, onDismiss: () -> Unit) :
    ReliefOverlayView(context, onDismiss) {

    init { title = "😄  A quick laugh" }

    private val jokes = listOf(
        "Why don't skeletons fight each other?\nThey don't have the guts.",
        "I told my computer I needed a break…\nnow it won't stop sending me KitKats.",
        "Why did the scarecrow win an award?\nHe was outstanding in his field.",
        "I'm reading a book about anti-gravity.\nIt's impossible to put down.",
        "What do you call fake spaghetti?\nAn impasta.",
        "Why did the bicycle fall over?\nIt was two-tired.",
        "I would tell you a chemistry joke…\nbut I know I wouldn't get a reaction.",
        "What do you call a bear with no teeth?\nA gummy bear.",
        "Why don't eggs tell jokes?\nThey'd crack each other up.",
        "I used to play piano by ear…\nnow I use my hands.",
        "What did the ocean say to the beach?\nNothing, it just waved.",
        "Why did the math book look so sad?\nIt had too many problems.",
    )
    private var idx = 0

    private val text = TextPaint(TextPaint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFEAEFF5.toInt()   // StaticLayout handles centering via setAlignment
    }
    private val hint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6E7480.toInt(); textAlign = android.graphics.Paint.Align.CENTER
    }

    override fun onShow() { idx = (idx + 1) % jokes.size }

    override fun drawContent(canvas: Canvas, w: Float, h: Float) {
        text.textSize = w * 0.062f
        val pad = w * 0.12f
        val layout = StaticLayout.Builder
            .obtain(jokes[idx], 0, jokes[idx].length, text, (w - 2 * pad).toInt())
            .setAlignment(Layout.Alignment.ALIGN_CENTER)
            .setLineSpacing(h * 0.012f, 1f)
            .build()
        val cy = (contentTop(h) + contentBottom(h)) / 2f - layout.height / 2f
        canvas.save()
        canvas.translate(pad, cy)
        layout.draw(canvas)
        canvas.restore()

        hint.textSize = w * 0.04f
        canvas.drawText("tap for another", w / 2f, contentBottom(h) - h * 0.01f, hint)
    }

    override fun onContentTouch(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP) {
            idx = (idx + 1) % jokes.size
            invalidate()
        }
        return true
    }
}
