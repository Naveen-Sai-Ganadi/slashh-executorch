package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.AudioConfig
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.StressPipeline
import ai.slashh.audio.StressClassifier
import ai.slashh.ui.CalmCue
import ai.slashh.ui.Calibration
import ai.slashh.relief.Prefs
import ai.slashh.relief.ReliefNotifier
import ai.slashh.relief.ReliefSelector
import ai.slashh.relief.ReliefType
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var prefs: Prefs
    private val calmCue = CalmCue()
    private var capture: AudioCapture? = null
    private var classifier: StressClassifier? = null
    private lateinit var pipeline: StressPipeline

    private var cueDriven = false
    private var reliefOpen = false
    private var activeReliefType: String? = null
    private var calibrationOpen = false

    // Background-monitor mirror: the StressMonitorService owns the mic and scores
    // on the NPU out-of-process; we poll its live reading onto the web gauge.
    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var mirroring = false
    private var mirrorStressed = false

    // State properties for JS sync
    private var stressScore: Float = 5f
    private var inferenceLatency: Long = 12
    private var lastOutputStr: String = "0.31"
    private var backendType: String = "CPU (XNNPACK)"
    private var fastRpcStatus: String = "checking"
    private var modelStatus: String = "loaded"
    private var loadedModelName: String = ""

    // Cooldown for notifications to avoid spamming
    private var lastNotificationTime: Long = 0L

    // Calibration state — collects BOTH model raw scores AND energy per window
    private var calibratingStep = -1
    private val calmScores = ArrayList<Float>()
    private val calmEnergy = ArrayList<Float>()
    private val stressedScores = ArrayList<Float>()
    private val stressedEnergy = ArrayList<Float>()

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        sendStateToWeb()
        if (granted) { if (calibrationOpen) startListening() else startMonitor() }
    }

    private val askNotif = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) {
        sendStateToWeb()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = Prefs(this)

        // Setup WebView
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            WebView.setWebContentsDebuggingEnabled(true)
        }

        webView = WebView(this).apply {
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = true
                allowContentAccess = true
                allowFileAccessFromFileURLs = true
                allowUniversalAccessFromFileURLs = true
            }
            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    Log.i("Slashh", "WebView page finished: $url")
                    sendStateToWeb()
                }
                override fun onPageStarted(view: WebView?, url: String?, favicon: android.graphics.Bitmap?) {
                    super.onPageStarted(view, url, favicon)
                    Log.i("Slashh", "WebView page started: $url")
                }
            }
            webChromeClient = object : android.webkit.WebChromeClient() {
                override fun onConsoleMessage(consoleMessage: android.webkit.ConsoleMessage?): Boolean {
                    consoleMessage?.let {
                        val level = when (it.messageLevel()) {
                            android.webkit.ConsoleMessage.MessageLevel.ERROR -> Log.ERROR
                            android.webkit.ConsoleMessage.MessageLevel.WARNING -> Log.WARN
                            else -> Log.INFO
                        }
                        Log.println(level, "SlashhWebView", "${it.message()} -- From line ${it.lineNumber()} of ${it.sourceId()}")
                    }
                    return true
                }
            }
            addJavascriptInterface(SlashhWebBridge(this@MainActivity), "AndroidBridge")
        }
        setContentView(webView)
        webView.loadUrl("file:///android_asset/www/index.html")

        // Load classifier (CPU, QNN/NPU or fallback mock)
        classifier = loadClassifier()
        if (classifier == null) {
            modelStatus = "missing"
            backendType = "Demo Fallback"
            fastRpcStatus = "unavailable"
            classifier = StressClassifier { _ ->
                // Simulated voice stress score
                (0.15f + 0.70f * Math.random().toFloat())
            }
        }
        
        pipeline = StressPipeline(classifier!!)

        // Apply any per-user calibration (signal choice + anchors)
        if (prefs.calibrated) {
            pipeline.useModelSignal = prefs.useModel
            prefs.calmAnchor?.let { pipeline.calmAnchor = it }
            prefs.stressAnchor?.let { pipeline.stressAnchor = it }
        }

        ReliefNotifier.ensureChannel(this)
        requestNotificationPermission()

        handleIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            if (calibrationOpen) startListening()   // in-app capture for calibration
            else startMonitor()                      // always-on background NPU monitor + mirror
        } else {
            // Ask up front so the always-on monitor can auto-start once granted.
            askMic.launch(Manifest.permission.RECORD_AUDIO)
        }
        sendStateToWeb()
    }

    override fun onPause() {
        super.onPause()
        // Stop the UI mirror, but KEEP the background monitor running — the whole
        // point is that on-device monitoring continues with the app closed.
        stopMirror()
        capture?.stop()
        capture = null
        pipeline.reset()
        calmCue.reset()
        reliefOpen = false
        activeReliefType = null
        sendStateToWeb()
    }

    fun startListening() {
        if (capture != null) return
        capture = AudioCapture { window ->
            val now = System.currentTimeMillis()
            val start = System.currentTimeMillis()
            val state = pipeline.onWindow(window)
            val end = System.currentTimeMillis()

            inferenceLatency = end - start
            stressScore = state.rawScore?.let { it * 100f } ?: stressScore
            lastOutputStr = state.rawScore?.let { "%.2f".format(it) } ?: lastOutputStr

            if (calibratingStep != -1) {
                // Calibrating: feed voiced model raw scores AND vocal energy
                val r = state.rawScore
                if (state.voiced && r != null) {
                    val energy = pipeline.vadRms.toFloat()
                    if (calibratingStep == 0) {
                        calmScores.add(r)
                        calmEnergy.add(energy)
                        if (calmScores.size >= 6) {
                            calibratingStep = -1
                        }
                    } else if (calibratingStep == 1) {
                        stressedScores.add(r)
                        stressedEnergy.add(energy)
                        if (stressedScores.size >= 6) {
                            calibratingStep = -1
                            try {
                                val res = Calibration.compute(
                                    calmScores, calmEnergy,
                                    stressedScores, stressedEnergy
                                )
                                prefs.saveCalibration(res.useModel, res.calmAnchor, res.stressAnchor)
                                pipeline.useModelSignal = res.useModel
                                pipeline.calmAnchor = res.calmAnchor
                                pipeline.stressAnchor = res.stressAnchor
                                pipeline.reset()
                                Log.i("Slashh", "Calibrated useModel=${res.useModel} calm=${res.calmAnchor} stress=${res.stressAnchor}")
                            } catch (e: Exception) {
                                Log.e("Slashh", "Calibration error", e)
                            }
                        }
                    }
                }
            } else {
                val show = calmCue.onState(state.stressed, now)
                runOnUiThread {
                    when {
                        calmCue.justTriggered -> fireRelief(cueDriven = true)
                        !show && cueDriven -> dismissRelief()
                    }
                }
            }
            runOnUiThread { sendStateToWeb() }
            Log.i("SlashhVAD", "voiced=${state.voiced} rms=%.5f zcr=%.3f floor=%.5f raw=${state.rawScore} ema=${state.level}".format(pipeline.vadRms, pipeline.vadZcr, pipeline.vadFloor))
        }.also { it.start() }
        sendStateToWeb()
    }

    fun stopListening() {
        capture?.stop()
        capture = null
        pipeline.reset()
        calmCue.reset()
        sendStateToWeb()
    }

    /** Auto-start the always-on background NPU monitor: a foreground service owns
     *  the mic and scores WavLM on the Hexagon NPU out-of-process via the shell
     *  helper (CPU StressNet if the helper isn't live). The app only MIRRORS its
     *  live reading onto the gauge — it never captures the mic in parallel. */
    private fun startMonitor() {
        capture?.stop(); capture = null      // never hold the mic alongside the service
        StressMonitorService.start(this)
        startMirror()
    }

    private fun startMirror() {
        if (mirroring) return
        mirroring = true
        mirrorStressed = false
        mainHandler.post(mirrorTick)
    }

    private fun stopMirror() {
        mirroring = false
        mainHandler.removeCallbacks(mirrorTick)
    }

    /** Poll the service's latest NPU reading (~400 ms) and reflect it on the gauge. */
    private val mirrorTick = object : Runnable {
        override fun run() {
            if (!mirroring) return
            StressMonitorService.lastState?.let { st ->
                stressScore = ((st.level ?: st.rawScore ?: 0f) * 100f).coerceIn(0f, 100f)
                st.rawScore?.let { lastOutputStr = "%.2f".format(it) }
                backendType = StressMonitorService.backend
                modelStatus = "loaded"
                fastRpcStatus = if (StressMonitorService.backend.contains("NPU")) "ok" else "checking"
                // Open the in-app relief overlay on sustained stress while visible;
                // the service itself raises the (background) notification.
                if (st.stressed && !mirrorStressed && !reliefOpen) fireRelief(cueDriven = true, notify = false)
                else if (!st.stressed && mirrorStressed && cueDriven) dismissRelief()
                mirrorStressed = st.stressed
                sendStateToWeb()
            }
            mainHandler.postDelayed(this, 400)
        }
    }

    fun toggleListening() {
        if (capture != null) stopListening() else startListening()
    }

    fun requestMicPermission() {
        askMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            askNotif.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    fun setSensitivity(sensitivity: Int) {
        // Map 10-100 sensitivity slider to pipeline threshold offsets
        // Higher sensitivity => lower enter threshold
        val baseThreshold = 0.55f
        val sensOffset = (sensitivity - 55f) / 100f * 0.30f
        pipeline.enterThreshold = (baseThreshold - sensOffset).coerceIn(0.15f, 0.90f)
        pipeline.releaseThreshold = (pipeline.enterThreshold - 0.12f).coerceIn(0.10f, pipeline.enterThreshold - 0.03f)
        Log.i("Slashh", "Set sensitivity enter=${pipeline.enterThreshold} release=${pipeline.releaseThreshold}")
    }

    fun setDemoMode(enabled: Boolean) {
        // Demo mode on Kotlin side can trigger fake high stress to demo relief
        if (enabled) {
            stressScore = 85f
            fireRelief(cueDriven = false)
        } else {
            stressScore = 5f
            dismissRelief()
        }
    }

    fun startCalibration() {
        calibratingStep = -1
        calmScores.clear()
        calmEnergy.clear()
        stressedScores.clear()
        stressedEnergy.clear()
        calibrationOpen = true
        // Calibration needs the in-app mic (model rawScore + energy), so hand the
        // mic back from the background monitor for the duration of the flow. Brief
        // delay so the service releases the mic before in-app capture opens.
        stopMirror()
        StressMonitorService.stop(this)
        mainHandler.postDelayed({ if (calibrationOpen) startListening() }, 300)
        sendStateToWeb()
    }

    fun cancelCalibration() {
        calibratingStep = -1
        calmScores.clear()
        calmEnergy.clear()
        stressedScores.clear()
        stressedEnergy.clear()
        calibrationOpen = false
        stopListening()
        startMonitor()      // resume the always-on background monitor
        sendStateToWeb()
    }

    fun saveCalibration(useModel: Boolean, calmAnchor: Double, stressAnchor: Double) {
        prefs.saveCalibration(useModel, calmAnchor.toFloat(), stressAnchor.toFloat())
        pipeline.useModelSignal = useModel
        pipeline.calmAnchor = calmAnchor.toFloat()
        pipeline.stressAnchor = stressAnchor.toFloat()
        pipeline.reset()
        calibrationOpen = false
        stopListening()
        startMonitor()      // resume the always-on background monitor
        sendStateToWeb()
    }

    fun startRelief(type: String) {
        reliefOpen = true
        activeReliefType = type
        sendStateToWeb()
    }

    private fun fireRelief(cueDriven: Boolean, notify: Boolean = true) {
        val enabled = prefs.enabled()
        if (enabled.isEmpty()) return
        val (type, idx) = ReliefSelector.next(enabled, prefs.lastIndex)
        prefs.lastIndex = idx

        reliefOpen = true
        activeReliefType = type.key
        this.cueDriven = cueDriven

        // Cooldown notification to avoid spamming. Skipped (notify=false) when the
        // background monitor already owns the notification path.
        val now = System.currentTimeMillis()
        if (notify && prefs.notify && (now - lastNotificationTime > 30000L)) {
            ReliefNotifier.notify(this, type)
            lastNotificationTime = now
        }
        sendStateToWeb()
    }

    fun dismissRelief() {
        calmCue.dismiss(System.currentTimeMillis())
        reliefOpen = false
        activeReliefType = null
        cueDriven = false
        sendStateToWeb()
    }

    fun sendStateToWeb() {
        val isListening = (capture != null) || StressMonitorService.running
        val micPermission = if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) "granted" else "denied"
        val notifPermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) "granted" else "denied"
        } else {
            "granted"
        }
        val calDone = prefs.calibrated
        val signalType = if (pipeline.useModelSignal) "model" else "energy"

        val currentState = when {
            micPermission == "denied" -> "PERMISSION_REQUIRED"
            calibrationOpen -> {
                when (calibratingStep) {
                    0 -> "CALIBRATING_NORMAL"
                    1 -> "CALIBRATING_STRESSED"
                    else -> "CALIBRATING"
                }
            }
            reliefOpen -> "RELIEF_ACTIVE"
            isListening -> {
                if (stressScore > 75f) "STRESS_DETECTED" else "LISTENING"
            }
            loadedModelName.isEmpty() -> "MODEL_MISSING"
            else -> "READY"
        }

        val json = """
            {
                "state": "$currentState",
                "listening": $isListening,
                "stress": ${stressScore.toInt()},
                "latency": $inferenceLatency,
                "lastOutput": "$lastOutputStr",
                "backendType": "$backendType",
                "fastRpcStatus": "$fastRpcStatus",
                "modelStatus": "$modelStatus",
                "micPermissionState": "$micPermission",
                "notificationPermissionState": "$notifPermission",
                "calibration": {
                    "done": $calDone,
                    "signalType": "$signalType",
                    "calmAnchor": ${pipeline.calmAnchor},
                    "stressAnchor": ${pipeline.stressAnchor},
                    "calmCount": ${calmScores.size},
                    "stressCount": ${stressedScores.size},
                    "step": $calibratingStep
                },
                "reliefOpen": $reliefOpen,
                "activeReliefType": ${if (activeReliefType != null) "\"$activeReliefType\"" else "null"},
                "calibrationOpen": $calibrationOpen
            }
        """.trimIndent().replace("\n", "").replace("\r", "")

        webView.evaluateJavascript("if (typeof window.updateAndroidState === 'function') { window.updateAndroidState('$json'); }", null)
    }

    fun startRecording(step: Int) {
        calibratingStep = step
        if (step == 0) { calmScores.clear(); calmEnergy.clear() }
        if (step == 1) { stressedScores.clear(); stressedEnergy.clear() }
        sendStateToWeb()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val key = intent?.getStringExtra(ReliefNotifier.EXTRA_SHOW_RELIEF) ?: return
        ReliefType.fromKey(key)?.let {
            reliefOpen = true
            activeReliefType = it.key
            cueDriven = false
            sendStateToWeb()
        }
    }

    /**
     * Try QNN/NPU first, then CPU/XNNPACK. Falls back to null if nothing loads.
     * After loading, validates with a test forward() — a QNN .pte can load fine
     * but its first forward() fails where CDSP/NPU access is blocked (retail S25).
     * See docs/NPU-ON-DEVICE.md.
     */
    private fun loadClassifier(): StressClassifier? {
        val featLen = AudioConfig.N_MELS * AudioConfig.N_FRAMES
        for (name in listOf("stress_model_qnn.pte", "stress_model.pte")) {
            try {
                val c = ExecuTorchStressClassifier(copyAsset(name))
                // Validate it can actually RUN — a QNN .pte loads fine but its
                // first forward() fails where CDSP/NPU access is blocked (retail
                // S25). Only then is it safe to keep; otherwise fall back to CPU.
                c.score(FloatArray(featLen))
                loadedModelName = name
                modelStatus = "loaded"
                backendType = if (name.contains("qnn")) "Snapdragon NPU (QNN)" else "CPU (XNNPACK)"
                fastRpcStatus = if (name.contains("qnn")) "ok" else "checking"
                Log.i("Slashh", "model loaded: $name ($backendType)")
                return c
            } catch (e: Throwable) {
                Log.w("Slashh", "could not load/run $name (${e.message}); trying next")
            }
        }
        return null
    }

    private fun copyAsset(name: String): String {
        val outFile = File(filesDir, name)
        if (!outFile.exists()) {
            assets.open(name).use { input -> outFile.outputStream().use { input.copyTo(it) } }
        }
        return outFile.absolutePath
    }

    // ---- Local account (on-device) — backing for the JS login/signup screen.
    // All credential storage is local SharedPreferences (SHA-256, no network).
    fun bridgeHasAccount(): Boolean = prefs.hasAccount
    fun bridgeGetName(): String = prefs.name
    fun bridgeCreateAccount(name: String, email: String, password: String) =
        prefs.createAccount(name, email, password)
    fun bridgeCheckLogin(email: String, password: String): Boolean =
        prefs.checkLogin(email, password)
    fun bridgeIsOnboarded(): Boolean = prefs.onboarded
    fun bridgeSetOnboarded(done: Boolean) { prefs.onboarded = done }

    /** Dev bypass: create a dummy local account (if none exists) and mark onboarded
     *  so the auth screen is skipped entirely during testing. */
    fun bridgeDevBypassAuth(): Boolean {
        if (!prefs.hasAccount) {
            prefs.createAccount("Dev User", "dev@slashh.ai", "devpass")
        }
        prefs.onboarded = true
        return true
    }

    /** Run WavLmProbe on a background thread. Results appear in logcat under
     *  the "WavLMProbe" tag. The model file must be adb-pushed to the app's
     *  external files dir as wavlm_int8.pte before calling this. */
    fun bridgeRunNpuProbe() {
        val modelFile = java.io.File(getExternalFilesDir(null), "wavlm_int8.pte")
        Thread {
            Log.i("Slashh", "runNpuProbe: launching WavLmProbe against ${modelFile.absolutePath}")
            ai.slashh.audio.WavLmProbe.run(modelFile)
        }.start()
    }
    fun bridgeSetEnabled(csvKeys: String) {
        val types = csvKeys.split(",").mapNotNull { ReliefType.fromKey(it.trim()) }
        if (types.isNotEmpty()) prefs.setEnabled(types)
    }

    override fun onDestroy() {
        super.onDestroy()
        (classifier as? AutoCloseable)?.close()
    }
}

class SlashhWebBridge(private val activity: MainActivity) {
    @JavascriptInterface
    fun toggleListening() = activity.runOnUiThread { activity.toggleListening() }

    @JavascriptInterface
    fun startListening() = activity.runOnUiThread { activity.startListening() }

    @JavascriptInterface
    fun stopListening() = activity.runOnUiThread { activity.stopListening() }

    @JavascriptInterface
    fun requestMicPermission() = activity.runOnUiThread { activity.requestMicPermission() }

    @JavascriptInterface
    fun requestNotificationPermission() = activity.runOnUiThread { activity.requestNotificationPermission() }

    @JavascriptInterface
    fun setSensitivity(sensitivity: Int) = activity.runOnUiThread { activity.setSensitivity(sensitivity) }

    @JavascriptInterface
    fun setDemoMode(enabled: Boolean) = activity.runOnUiThread { activity.setDemoMode(enabled) }

    @JavascriptInterface
    fun startCalibration() = activity.runOnUiThread { activity.startCalibration() }

    @JavascriptInterface
    fun cancelCalibration() = activity.runOnUiThread { activity.cancelCalibration() }

    @JavascriptInterface
    fun saveCalibration(useModel: Boolean, calmAnchor: Double, stressAnchor: Double) =
        activity.runOnUiThread { activity.saveCalibration(useModel, calmAnchor, stressAnchor) }

    @JavascriptInterface
    fun startRelief(type: String) = activity.runOnUiThread { activity.startRelief(type) }

    @JavascriptInterface
    fun stopRelief() = activity.runOnUiThread { activity.dismissRelief() }

    @JavascriptInterface
    fun dismissRelief() = activity.runOnUiThread { activity.dismissRelief() }

    @JavascriptInterface
    fun getBackendStatus() = activity.runOnUiThread { activity.sendStateToWeb() }

    @JavascriptInterface
    fun startRecording(step: Int) = activity.runOnUiThread { activity.startRecording(step) }

    // Local account (on-device). Value-returning methods must run synchronously
    // on the JS-bridge thread (NOT runOnUiThread) so the result reaches JS; Prefs
    // is SharedPreferences-backed and thread-safe.
    @JavascriptInterface
    fun hasAccount(): Boolean = activity.bridgeHasAccount()

    @JavascriptInterface
    fun getName(): String = activity.bridgeGetName()

    @JavascriptInterface
    fun createAccount(name: String, email: String, password: String) =
        activity.bridgeCreateAccount(name, email, password)

    @JavascriptInterface
    fun checkLogin(email: String, password: String): Boolean =
        activity.bridgeCheckLogin(email, password)

    @JavascriptInterface
    fun isOnboarded(): Boolean = activity.bridgeIsOnboarded()

    @JavascriptInterface
    fun setEnabled(csvKeys: String) = activity.bridgeSetEnabled(csvKeys)

    @JavascriptInterface
    fun setOnboarded(done: Boolean) = activity.bridgeSetOnboarded(done)

    @JavascriptInterface
    fun devBypassAuth(): Boolean = activity.bridgeDevBypassAuth()

    @JavascriptInterface
    fun runNpuProbe() = activity.bridgeRunNpuProbe()
}
