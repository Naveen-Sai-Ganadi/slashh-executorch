package ai.slashh.relief

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.MotionEvent
import kotlin.math.min

/**
 * A simple tic-tac-toe relief: you are X, the phone plays O (takes the win,
 * else blocks, else center/random). Tap a cell to play; tap anywhere after a
 * result to play again. Dismiss via the shared "I feel better" pill.
 */
class TicTacToeView(context: Context, onDismiss: () -> Unit) :
    ReliefOverlayView(context, onDismiss) {

    init { title = "⭕  Tic-tac-toe" }

    private val board = CharArray(9) { ' ' }
    private var over = false
    private var status = "Your turn — you're X"

    private val grid = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF2C3442.toInt(); strokeWidth = 8f; style = Paint.Style.STROKE
    }
    private val xPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF4FC3F7.toInt(); strokeWidth = 18f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val oPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFF06292.toInt(); strokeWidth = 18f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFB7C0CC.toInt(); textAlign = Paint.Align.CENTER
    }

    private var left = 0f; private var top = 0f; private var cell = 0f

    override fun onShow() {
        board.fill(' '); over = false; status = "Your turn — you're X"
    }

    override fun drawContent(canvas: Canvas, w: Float, h: Float) {
        val size = min(w * 0.8f, contentBottom(h) - contentTop(h) - h * 0.06f)
        cell = size / 3f
        left = (w - size) / 2f
        top = contentTop(h) + h * 0.04f

        statusPaint.textSize = w * 0.05f
        canvas.drawText(status, w / 2f, contentTop(h) - h * 0.01f, statusPaint)

        // grid lines
        for (i in 1..2) {
            canvas.drawLine(left + cell * i, top, left + cell * i, top + size, grid)
            canvas.drawLine(left, top + cell * i, left + size, top + cell * i, grid)
        }
        // marks
        for (i in 0..8) {
            val cx = left + (i % 3) * cell + cell / 2f
            val cy = top + (i / 3) * cell + cell / 2f
            val r = cell * 0.28f
            when (board[i]) {
                'X' -> {
                    canvas.drawLine(cx - r, cy - r, cx + r, cy + r, xPaint)
                    canvas.drawLine(cx + r, cy - r, cx - r, cy + r, xPaint)
                }
                'O' -> canvas.drawCircle(cx, cy, r, oPaint)
            }
        }
    }

    override fun onContentTouch(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_UP) return true
        if (over) { onShow(); invalidate(); return true }
        val col = ((event.x - left) / cell).toInt()
        val row = ((event.y - top) / cell).toInt()
        if (col !in 0..2 || row !in 0..2) return true
        val idx = row * 3 + col
        if (board[idx] != ' ') return true

        board[idx] = 'X'
        if (resolve('X', "You win! 🎉")) { invalidate(); return true }
        aiMove()
        resolve('O', "Phone wins — rematch?")
        invalidate()
        return true
    }

    private fun aiMove() {
        val empty = (0..8).filter { board[it] == ' ' }
        if (empty.isEmpty()) return
        // take a win, else block, else center, else first empty
        val move = winningMove('O') ?: winningMove('X') ?: 4.takeIf { board[4] == ' ' } ?: empty.first()
        board[move] = 'O'
    }

    private fun winningMove(p: Char): Int? {
        for (i in 0..8) if (board[i] == ' ') {
            board[i] = p
            val win = winner() == p
            board[i] = ' '
            if (win) return i
        }
        return null
    }

    private fun resolve(p: Char, winMsg: String): Boolean {
        if (winner() == p) { status = winMsg; over = true; return true }
        if (board.none { it == ' ' }) { status = "Draw — tap to replay"; over = true; return true }
        if (p == 'O' && !over) status = "Your turn — you're X"
        return false
    }

    private fun winner(): Char {
        val lines = arrayOf(
            intArrayOf(0, 1, 2), intArrayOf(3, 4, 5), intArrayOf(6, 7, 8),
            intArrayOf(0, 3, 6), intArrayOf(1, 4, 7), intArrayOf(2, 5, 8),
            intArrayOf(0, 4, 8), intArrayOf(2, 4, 6),
        )
        for (l in lines) {
            val a = board[l[0]]
            if (a != ' ' && a == board[l[1]] && a == board[l[2]]) return a
        }
        return ' '
    }
}
