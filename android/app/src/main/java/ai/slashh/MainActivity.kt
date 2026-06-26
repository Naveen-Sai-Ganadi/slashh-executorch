package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.StressPipeline
import ai.slashh.relief.ColorToyView
import ai.slashh.relief.JokesView
import ai.slashh.relief.OnboardingView
import ai.slashh.relief.Prefs
import ai.slashh.relief.ReliefNotifier
import ai.slashh.relief.ReliefScreen
import ai.slashh.relief.ReliefSelector
import ai.slashh.relief.ReliefType
import ai.slashh.relief.SoundsView
import ai.slashh.relief.TicTacToeView
import ai.slashh.ui.BreathOverlayView
import ai.slashh.ui.CalmCue
import ai.slashh.ui.Meter
import ai.slashh.ui.StressMeterView
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.FrameLayout
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

/**
 * End-to-end wiring: permission → load `.pte` → capture → pipeline → meter (M8)
 * → **personalized relief** (M9+): on sustained stress, [CalmCue] decides *when*,
 * [ReliefSelector] picks *which* relief from the user's local preferences, and we
 * show it in-app AND post a local notification. First launch shows onboarding.
 *
 * Everything is on-device; there is no network code anywhere in this path.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var meter: StressMeterView
    private lateinit var root: FrameLayout
    private lateinit var prefs: Prefs
    private val calmCue = CalmCue()
    private var capture: AudioCapture? = null
    private var classifier: ExecuTorchStressClassifier? = null
    private lateinit var pipeline: StressPipeline

    private val reliefs = LinkedHashMap<ReliefType, ReliefScreen>()
    private var current: ReliefScreen? = null
    /** true when the current relief was auto-triggered by stress (so the loop may
     *  auto-hide it when stress clears); false when opened manually from a
     *  notification/deep-link (stays until the user dismisses). */
    private var cueDriven = false
    private var onboarding: OnboardingView? = null

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startListening() else meter.render(Meter.needMic()) }

    private val askNotif = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* relief still works in-app if denied */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)
        meter = StressMeterView(this)
        root = FrameLayout(this).apply { addView(meter) }

        // Build every relief as a hidden overlay; the router shows one on demand.
        val dismiss = { dismissCurrent() }
        reliefs[ReliefType.BREATHING] = BreathOverlayView(this, dismiss).gone()
        reliefs[ReliefType.TIC_TAC_TOE] = TicTacToeView(this, dismiss).gone()
        reliefs[ReliefType.SOUNDS] = SoundsView(this, dismiss).gone()
        reliefs[ReliefType.JOKES] = JokesView(this, dismiss).gone()
        reliefs[ReliefType.COLOR] = ColorToyView(this, dismiss).gone()
        reliefs.values.forEach { root.addView(it as android.view.View) }
        setContentView(root)

        val modelPath = copyAsset("stress_model.pte")
        classifier = ExecuTorchStressClassifier(modelPath)
        pipeline = StressPipeline(classifier!!)

        ReliefNotifier.ensureChannel(this)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) askNotif.launch(Manifest.permission.POST_NOTIFICATIONS)

        if (!prefs.onboarded) showOnboarding()
        handleIntent(intent)
    }

    private fun <T : android.view.View> T.gone(): T { visibility = android.view.View.GONE; return this }

    private fun showOnboarding() {
        val ob = OnboardingView(this) { selected ->
            prefs.setEnabled(selected)
            prefs.onboarded = true
            onboarding?.let { root.removeView(it) }
            onboarding = null
        }
        onboarding = ob
        root.addView(ob)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) startListening() else askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    override fun onPause() {
        super.onPause()
        capture?.stop()
        capture = null
        pipeline.reset()
        calmCue.reset()
        hideCurrent()
    }

    private fun startListening() {
        if (capture != null) return
        capture = AudioCapture { window ->
            val now = System.currentTimeMillis()
            val state = pipeline.onWindow(window)
            val model = Meter.from(state)
            val show = calmCue.onState(state.stressed, now)
            runOnUiThread {
                meter.render(model)
                when {
                    calmCue.justTriggered -> triggerRelief()
                    !show && cueDriven -> hideCurrent()
                }
            }
            Log.d("Slashh", "voiced=${state.voiced} raw=${state.rawScore} level=${state.level} stressed=${state.stressed}")
        }.also { it.start() }
    }

    /** Pick a relief from the user's preferences, show it in-app + notify. */
    private fun triggerRelief() {
        val enabled = prefs.enabled()
        if (enabled.isEmpty()) return
        val (type, idx) = ReliefSelector.next(enabled, prefs.lastIndex)
        prefs.lastIndex = idx
        showRelief(type)
        cueDriven = true
        if (prefs.notify) ReliefNotifier.notify(this, type)
    }

    private fun showRelief(type: ReliefType) {
        current?.hide()
        current = reliefs[type]
        cueDriven = false      // manual unless triggerRelief() flips it
        current?.show()
    }

    private fun dismissCurrent() {
        calmCue.dismiss(System.currentTimeMillis())
        hideCurrent()
    }

    private fun hideCurrent() {
        current?.hide()
        current = null
        cueDriven = false
    }

    /** Notification tap: jump straight to the suggested relief. */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val key = intent?.getStringExtra(ReliefNotifier.EXTRA_SHOW_RELIEF) ?: return
        ReliefType.fromKey(key)?.let { showRelief(it) }
    }

    /** ExecuTorch loads from a filesystem path; copy the bundled asset out once. */
    private fun copyAsset(name: String): String {
        val outFile = File(filesDir, name)
        if (!outFile.exists()) {
            assets.open(name).use { input -> outFile.outputStream().use { input.copyTo(it) } }
        }
        return outFile.absolutePath
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
