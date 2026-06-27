package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.StressClassifier
import ai.slashh.audio.StressPipeline
import ai.slashh.audio.WavLmStressClassifier
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
import android.system.Os
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
    private var wavlm: WavLmStressClassifier? = null
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

    // While the background monitor owns the mic, the in-app capture loop is stopped,
    // so the gauge has nothing to draw and falls back to its idle "Listening…" dash.
    // This poller mirrors the service's latest NPU reading onto the meter (~4 Hz)
    // while the app is in the foreground, so an open app shows the live level/label
    // instead of a dash. It runs only between onResume/onPause and only while the
    // monitor is running; the service still owns scoring + notifications.
    private val uiHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val monitorMeterTick = object : Runnable {
        override fun run() {
            if (StressMonitorService.running) {
                // The monitor owns the mic — make sure we're not also capturing
                // in-app (two concurrent mic streams from one app is unreliable).
                capture?.let { it.stop(); capture = null }
                val state = StressMonitorService.lastState
                meter.render(if (state != null) Meter.from(state) else Meter.monitoring())
            }
            // Keep ticking while the activity is resumed; do NOT self-terminate on a
            // transient !running. StressMonitorService.start() is async — the
            // foreground service's onStartCommand flips `running` true a few ms AFTER
            // we post this poller, so the first tick right after tapping Start can
            // still observe running=false. Killing the poller there left the gauge
            // frozen on its idle dash even though the monitor was scoring. The poller
            // is stopped explicitly in onPause.
            uiHandler.postDelayed(this, 250L)
        }
    }
    private fun startMonitorMeter() {
        uiHandler.removeCallbacks(monitorMeterTick)
        uiHandler.post(monitorMeterTick)
    }
    private fun stopMonitorMeter() {
        uiHandler.removeCallbacks(monitorMeterTick)
    }

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startListening() else meter.render(Meter.needMic()) }

    private val askNotif = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* relief still works in-app if denied */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        configureQnnSkelSearchPath()
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

        pipeline = buildPipeline()
        // apply any per-user calibrated thresholds
        prefs.enterThreshold?.let { pipeline.enterThreshold = it }
        prefs.releaseThreshold?.let { pipeline.releaseThreshold = it }

        ReliefNotifier.ensureChannel(this)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) askNotif.launch(Manifest.permission.POST_NOTIFICATIONS)

        meter.onSimulate = { fireRelief(cueDriven = false) }
        meter.onCalibrate = { showCalibration() }
        meter.onToggleMonitor = { toggleMonitor() }
        meter.monitorActive = StressMonitorService.running
        gateAuth()
        handleIntent(intent)
    }

    private fun <T : android.view.View> T.gone(): T { visibility = android.view.View.GONE; return this }

    /**
     * Point the fastrpc DSP loader at our extracted native libs so it can load the
     * Hexagon skel (`libQnnHtpV79Skel.so`) onto the cDSP.
     *
     * The QNN HTP backend creates its device by loading that skel via the CPU-side
     * fastrpc shim (`libcdsprpc.so`), which locates the skel by searching
     * `ADSP_LIBRARY_PATH`. Android's default value lists only system DSP dirs
     * (`/vendor/lib/rfsa/adsp`, …) — never an app's `nativeLibraryDir`. With legacy
     * packaging the skel *is* extracted to our lib dir, but it's off the search
     * path, so QNN aborts at `QnnDevice_create` with
     * `QnnDsp <E> Failed to load skel, error: 4000`. Prepending our lib dir
     * (entries are `;`-separated per the QNN/fastrpc convention, not `:`) lets the
     * retail-device cDSP find and load it. Must run before the first `Module.load`.
     */
    private fun configureQnnSkelSearchPath() {
        val libDir = applicationInfo.nativeLibraryDir
        val path = "$libDir;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/lib64/rfsa/adsp;/dsp"
        try {
            Os.setenv("ADSP_LIBRARY_PATH", path, true)
            Log.i("Slashh", "ADSP_LIBRARY_PATH=$path")
        } catch (e: Throwable) {
            Log.w("Slashh", "could not set ADSP_LIBRARY_PATH: ${e.message}")
        }
    }

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
        meter.monitorActive = StressMonitorService.running
        // Always arm the mirror poller while resumed. It renders the service's live
        // NPU reading when the monitor is running and no-ops otherwise (leaving the
        // in-app capture loop to own the gauge). Arming it unconditionally makes the
        // gauge self-heal regardless of the order in which "app foreground" and
        // "monitor running" become true — e.g. a START_STICKY service restart that
        // flips `running` true a moment AFTER onResume would otherwise never get a
        // poller, leaving the gauge stuck on its idle dash.
        startMonitorMeter()
        // When the background monitor owns the mic, don't also capture in-app (two
        // concurrent MIC streams from one app is unreliable). The service does the
        // scoring + notifying; the poller above just mirrors it onto the gauge.
        if (StressMonitorService.running) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) startListening() else askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    /** Start/stop the always-on background stress monitor (foreground service). */
    private fun toggleMonitor() {
        if (StressMonitorService.running) {
            StressMonitorService.stop(this)
            meter.monitorActive = false
            stopMonitorMeter()
            // resume the in-app meter loop now that the mic is free
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED
            ) startListening()
            return
        }
        // need the mic before we can hand it to the service
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) { askMic.launch(Manifest.permission.RECORD_AUDIO); return }
        // free the in-app mic, then start the service to own it
        capture?.stop(); capture = null; pipeline.reset(); calmCue.reset()
        StressMonitorService.start(this)
        meter.monitorActive = true
        startMonitorMeter()
    }

    override fun onPause() {
        super.onPause()
        stopMonitorMeter()
        capture?.stop()
        capture = null
        pipeline.reset()
        calmCue.reset()
        hideCurrent()
    }

    private fun startListening() {
        if (capture != null) return
        if (StressMonitorService.running) return   // service owns the mic
        capture = AudioCapture { window ->
            // Scoring runs on the capture thread; a scorer failure (e.g. an
            // ExecuTorch delegate error on this device) must degrade to "no
            // reading" rather than throwing on the audio thread and crashing the
            // whole app. The background monitor's worker uses the same guard.
            try {
                val now = System.currentTimeMillis()
                val state = pipeline.onWindow(window)
                val cal = calibrationView
                if (cal != null) {
                    // calibrating: feed voiced scores, pause the meter/relief loop
                    val r = state.rawScore
                    if (state.voiced && r != null && cal.isCollecting()) {
                        runOnUiThread { cal.feedScore(r) }
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
                Log.d("Slashh", "voiced=${state.voiced} raw=${state.rawScore} level=${state.level} stressed=${state.stressed}")
            } catch (e: Throwable) {
                Log.w("Slashh", "in-app scoring error (degrading to no reading): ${e.message}")
            }
        }.also { it.start() }
    }

    /** Show the on-device calibration flow; on finish, persist + apply thresholds. */
    private fun showCalibration() {
        if (calibrationView != null) return
        hideCurrent()
        val view = CalibrationView(
            this,
            onDone = { enter, release ->
                prefs.saveThresholds(enter, release)
                pipeline.enterThreshold = enter
                pipeline.releaseThreshold = release
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
     * Build the scoring pipeline, preferring the **WavLM teacher** when present.
     *
     * The WavLM QNN `.pte` is hundreds of MB — too large to bundle in the APK — so
     * it ships out-of-band in the app's external files dir:
     * `adb push … /sdcard/Android/data/ai.slashh/files/teacher_wavlm_broad_qnn.pte`.
     * When that file exists we feed raw 16 kHz windows straight to the on-NPU graph
     * (it normalizes + extracts features + pools + scores in-graph) via
     * [WavLmStressClassifier], and the host-side log-mel + StressNet path is
     * skipped (the [StressClassifier] arg is an unused placeholder). If the WavLM
     * `.pte` is absent or fails to load, we fall back to the bundled StressNet
     * ([loadClassifier], QNN if `stress_model_qnn.pte` is present else CPU).
     */
    private fun buildPipeline(): StressPipeline {
        val wavlmFile = File(getExternalFilesDir(null), "teacher_wavlm_broad_qnn.pte")
        if (wavlmFile.exists()) {
            try {
                val w = WavLmStressClassifier(wavlmFile.absolutePath)
                wavlm = w
                Log.i("Slashh", "scorer: WavLM teacher (raw waveform, QNN) <- ${wavlmFile.name}")
                return StressPipeline(StressClassifier { 0f }, rawScorer = w)
            } catch (e: Throwable) {
                Log.w("Slashh", "WavLM .pte present but failed to load (${e.message}); using StressNet")
            }
        }
        val c = loadClassifier()
        classifier = c
        Log.i("Slashh", "scorer: StressNet (log-mel features)")
        return StressPipeline(c)
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
        val candidates = listOf("stress_model_qnn.pte", "stress_model.pte")
        val present = runCatching { assets.list("")?.toSet() ?: emptySet() }.getOrDefault(emptySet())
        for (name in candidates) {
            if (name !in present) continue
            try {
                val c = ExecuTorchStressClassifier(copyAsset(name))
                Log.i("Slashh", "model loaded: $name (${if (name.contains("qnn")) "NPU/QNN" else "CPU/XNNPACK"})")
                return c
            } catch (e: Throwable) {
                Log.w("Slashh", "could not load $name (${e.message}); trying next")
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
        wavlm?.close()
    }
}
