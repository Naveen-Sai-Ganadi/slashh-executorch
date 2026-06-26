package ai.slashh.relief

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.util.Patterns
import android.view.Gravity
import android.view.View
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Local sign-up / log-in (on-device only — no server, no network). Credentials
 * are hashed and stored in [Prefs]; this just gates the app and toggles between
 * "create account" and "log in". Clean, dark, professional.
 */
@SuppressLint("ViewConstructor")
class AuthView(
    context: Context,
    private val prefs: Prefs,
    private var signup: Boolean,
    private val onAuthed: () -> Unit,
) : FrameLayout(context) {

    private val col = LinearLayout(context).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER_HORIZONTAL
        setPadding(dp(28), dp(56), dp(28), dp(40))
    }
    private val bgPaint = Paint()

    init {
        setWillNotDraw(false)
        isClickable = true
        addView(ScrollView(context).apply {
            isFillViewport = true
            addView(col)
        })
        build()
    }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) {
        bgPaint.shader = LinearGradient(0f, 0f, 0f, h.toFloat(),
            0xFF14161D.toInt(), 0xFF05060A.toInt(), Shader.TileMode.CLAMP)
    }

    override fun onDraw(canvas: android.graphics.Canvas) {
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bgPaint)
    }

    private fun build() {
        col.removeAllViews()

        col.addV(wordmark("slashh"), 0)
        col.addV(muted("private, on-device stress relief", 14f), 6)
        col.addV(heading(if (signup) "Create your account" else "Welcome back"), 36)

        val name = if (signup) field("Name", InputType.TYPE_CLASS_TEXT or
            InputType.TYPE_TEXT_VARIATION_PERSON_NAME or InputType.TYPE_TEXT_FLAG_CAP_WORDS) else null
        name?.let { col.addV(it, 20) }
        val email = field("Email", InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS or InputType.TYPE_CLASS_TEXT)
        col.addV(email, if (signup) 12 else 20)
        val pass = field("Password", InputType.TYPE_TEXT_VARIATION_PASSWORD or InputType.TYPE_CLASS_TEXT)
        col.addV(pass, 12)

        val error = muted("", 13f).apply { setTextColor(0xFFE57373.toInt()); visibility = GONE }
        col.addV(error, 10)

        val cta = primary(if (signup) "Create account" else "Log in")
        col.addV(cta, 18)

        val toggle = link(
            if (signup) "Already have an account?  Log in"
            else "New here?  Create an account"
        )
        col.addV(toggle, 18)

        col.addV(muted("🔒  stored only on this device", 12.5f), 28)

        cta.setOnClickListener {
            val e = email.text.toString().trim()
            val p = pass.text.toString()
            val n = name?.text?.toString()?.trim() ?: ""
            val msg = validate(signup, n, e, p)
            if (msg != null) { error.text = msg; error.visibility = VISIBLE; return@setOnClickListener }
            if (signup) {
                prefs.createAccount(n, e, p)
                onAuthed()
            } else if (prefs.checkLogin(e, p)) {
                onAuthed()
            } else {
                error.text = "Incorrect email or password"; error.visibility = VISIBLE
            }
        }
        toggle.setOnClickListener { signup = !signup; build() }
    }

    private fun validate(signup: Boolean, name: String, email: String, pass: String): String? {
        if (signup && name.isEmpty()) return "Please enter your name"
        if (!Patterns.EMAIL_ADDRESS.matcher(email).matches()) return "Enter a valid email"
        if (pass.length < 4) return "Password must be at least 4 characters"
        return null
    }

    // ---- styled view builders ------------------------------------------------
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun LinearLayout.addV(v: View, topDp: Int) {
        val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT)
        lp.topMargin = dp(topDp)
        addView(v, lp)
    }

    private fun wordmark(t: String) = TextView(context).apply {
        text = t; setTextColor(Color.WHITE); textSize = 34f; gravity = Gravity.CENTER
        setTypeface(typeface, Typeface.BOLD)
    }

    private fun heading(t: String) = TextView(context).apply {
        text = t; setTextColor(0xFFE6EAF0.toInt()); textSize = 20f; gravity = Gravity.CENTER
        setTypeface(typeface, Typeface.BOLD)
    }

    private fun muted(t: String, size: Float) = TextView(context).apply {
        text = t; setTextColor(0xFF8A93A0.toInt()); textSize = size; gravity = Gravity.CENTER
    }

    private fun field(hintText: String, type: Int) = EditText(context).apply {
        hint = hintText; setHintTextColor(0xFF66707E.toInt()); setTextColor(Color.WHITE)
        textSize = 16f; inputType = type
        background = GradientDrawable().apply {
            cornerRadius = dp(16).toFloat(); setColor(0xFF121821.toInt()); setStroke(dp(1), 0xFF26303F.toInt())
        }
        setPadding(dp(18), dp(16), dp(18), dp(16))
    }

    private fun primary(label: String) = TextView(context).apply {
        text = label; setTextColor(Color.WHITE); textSize = 17f; gravity = Gravity.CENTER
        setTypeface(typeface, Typeface.BOLD)
        background = GradientDrawable().apply { cornerRadius = dp(28).toFloat(); setColor(0xFF2E7D5B.toInt()) }
        setPadding(0, dp(16), 0, dp(16)); isClickable = true; isFocusable = true
    }

    private fun link(label: String) = TextView(context).apply {
        text = label; setTextColor(0xFF7FD1A6.toInt()); textSize = 14f; gravity = Gravity.CENTER
        setPadding(0, dp(8), 0, dp(8)); isClickable = true; isFocusable = true
    }
}
