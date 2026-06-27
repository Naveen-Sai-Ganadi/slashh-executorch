package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.StressPipeline
import ai.slashh.relief.AuthView
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
import ai.slashh.ui.CalibrationView
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
    private var authView: AuthView? = null
    private var calibrationView: CalibrationView? = null

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

        classifier = loadClassifier()
        pipeline = StressPipeline(classifier!!)
        // apply any per-user calibration anchors (raw->stress mapping)
        prefs.calmAnchor?.let { pipeline.calmAnchor = it }
        prefs.stressAnchor?.let { pipeline.stressAnchor = it }

        ReliefNotifier.ensureChannel(this)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) askNotif.launch(Manifest.permission.POST_NOTIFICATIONS)

        meter.onSimulate = { fireRelief(cueDriven = false) }
        meter.onCalibrate = { showCalibration() }
        gateAuth()
        handleIntent(intent)
    }

    private fun <T : android.view.View> T.gone(): T { visibility = android.view.View.GONE; return this }

    /** Local login/signup gate, then onboarding on first run. */
    private fun gateAuth() {
        val auth = AuthView(this, prefs, signup = !prefs.hasAccount) {
            authView?.let { root.removeView(it) }
            authView = null
            if (!prefs.onboarded) showOnboarding()
        }
        authView = auth
        root.addView(auth)
    }

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
            val cal = calibrationView
            if (cal != null) {
                // calibrating: feed voiced vocal-intensity (energy) samples
                if (state.voiced && cal.isCollecting()) {
                    val energy = pipeline.vadRms.toFloat()
                    runOnUiThread { cal.feedScore(energy) }
                }
            } else {
                val model = Meter.from(state)
                val show = calmCue.onState(state.stressed, now)
                runOnUiThread {
                    meter.render(model)
                    when {
                        calmCue.justTriggered -> fireRelief(cueDriven = true)
                        !show && cueDriven -> hideCurrent()
                    }
                }
            }
            Log.i("SlashhVAD", "voiced=${state.voiced} rms=%.5f zcr=%.3f floor=%.5f raw=${state.rawScore} ema=${state.level}".format(pipeline.vadRms, pipeline.vadZcr, pipeline.vadFloor))
        }.also { it.start() }
    }

    /** Show the on-device calibration flow; on finish, persist + apply thresholds. */
    private fun showCalibration() {
        if (calibrationView != null) return
        hideCurrent()
        val view = CalibrationView(
            this,
            onDone = { calmAnchor, stressAnchor ->
                prefs.saveAnchors(calmAnchor, stressAnchor)
                pipeline.calmAnchor = calmAnchor
                pipeline.stressAnchor = stressAnchor
                pipeline.reset()
                calibrationView?.let { root.removeView(it) }
                calibrationView = null
            },
            onCancel = {
                calibrationView?.let { root.removeView(it) }
                calibrationView = null
            },
        )
        calibrationView = view
        root.addView(view)
    }

    /** Pick a relief from the user's preferences, show it in-app + notify.
     *  cueDriven=true → auto-hide when stress clears; false (simulate/demo) →
     *  stays until the user dismisses. */
    private fun fireRelief(cueDriven: Boolean) {
        val enabled = prefs.enabled()
        if (enabled.isEmpty()) return
        val (type, idx) = ReliefSelector.next(enabled, prefs.lastIndex)
        prefs.lastIndex = idx
        showRelief(type)
        this.cueDriven = cueDriven
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

    /**
     * Load the stress model, preferring the NPU build if present.
     *
     * Tries `stress_model_qnn.pte` (a QNN/Hexagon-delegated program) first, then
     * falls back to `stress_model.pte` (XNNPACK/CPU). This makes the NPU a
     * drop-in: at the event, add the QNN `.pte` to assets/ and ship the
     * QNN-enabled ExecuTorch AAR + Qualcomm runtime libs — no code change here.
     * If the QNN program/libs aren't available, Module.load throws and we fall
     * back to CPU. See docs/NPU-ON-DEVICE.md.
     */
    private fun loadClassifier(): ExecuTorchStressClassifier {
        // Try each asset directly (copyAsset opens it). Don't gate on
        // assets.list("") — it's unreliable on some OEM builds (e.g. this S25),
        // which made the app crash there even though the .pte ships in the APK.
        val featLen = ai.slashh.audio.AudioConfig.N_MELS * ai.slashh.audio.AudioConfig.N_FRAMES
        for (name in listOf("stress_model_qnn.pte", "stress_model.pte")) {
            try {
                val c = ExecuTorchStressClassifier(copyAsset(name))
                // Validate it can actually RUN — a QNN .pte loads fine but its
                // first forward() fails where CDSP/NPU access is blocked (retail
                // S25). Only then is it safe to keep; otherwise fall back to CPU.
                c.score(FloatArray(featLen))
                Log.i("Slashh", "model loaded: $name (${if (name.contains("qnn")) "NPU/QNN" else "CPU/XNNPACK"})")
                return c
            } catch (e: Throwable) {
                Log.w("Slashh", "could not load/run $name (${e.message}); trying next")
            }
        }
        error("no loadable stress model in assets")
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
