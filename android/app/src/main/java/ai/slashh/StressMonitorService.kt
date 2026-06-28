package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.LogMel
import ai.slashh.audio.NpuHelperScorer
import ai.slashh.audio.RawWaveScorer
import ai.slashh.audio.StressClassifier
import ai.slashh.audio.StressPipeline
import ai.slashh.audio.WavLmCoordinator
import ai.slashh.runtime.FusionScorer
import ai.slashh.runtime.NpuWhisperTranscriber
import ai.slashh.runtime.TextStressClassifier
import ai.slashh.runtime.Transcriber
import ai.slashh.runtime.TranscriptionCoordinator
import ai.slashh.relief.Prefs
import ai.slashh.relief.ReliefNotifier
import ai.slashh.relief.ReliefSelector
import ai.slashh.ui.CalmCue
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File
import java.util.concurrent.atomic.AtomicReference

/**
 * Always-on background stress monitor (the user's core ask: "I want my model
 * running in the background continuously irrespective of the app, and when stress
 * is detected raise a notification").
 *
 * It runs as a **foreground service (type microphone)** so Android keeps the mic
 * pipeline alive while the app UI is closed. The scoring itself happens on the
 * **Hexagon NPU out of process** via [NpuHelperScorer] → `npu_helper.sh` (the app
 * domain is SELinux-blocked from the cDSP on a retail S25, so the in-app QNN
 * delegate can't be used here). When sustained stress crosses the latch we fire a
 * high-priority **heads-up notification** ([ReliefNotifier]) that taps to reopen
 * the app — exactly the "Notification only" trigger the user chose.
 *
 * THREADING: the [AudioCapture] callback runs on the capture thread and must stay
 * light, but an NPU round-trip is ~1.2 s — longer than the 1 s hop. So the callback
 * only stashes the most-recent window in [latest] (latest-wins; stale windows are
 * dropped, which is the right behavior for a live monitor) and a dedicated worker
 * drains it, scores through the [StressPipeline], and drives [CalmCue]. Capture is
 * never blocked on the NPU.
 */
class StressMonitorService : Service() {

    private val latest = AtomicReference<FloatArray?>(null)
    @Volatile private var alive = false
    private var capture: AudioCapture? = null
    private var worker: Thread? = null
    private lateinit var prefs: Prefs
    private val calmCue = CalmCue()
    private var cpuFallback: ExecuTorchStressClassifier? = null

    // Whisper transcription + text-stress fusion (the second feature). Built best-effort on
    // the worker thread; absent assets / a down helper degrade honestly to audio-only.
    private var coordinator: TranscriptionCoordinator? = null
    private var textClassifier: TextStressClassifier? = null
    private var fusionScorer: FusionScorer? = null
    private var wavlmCoordinator: WavLmCoordinator? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (alive) return START_STICKY     // already running; ignore re-starts

        prefs = Prefs(this)
        ReliefNotifier.ensureChannel(this)
        ensureMonitorChannel()
        startForegroundCompat()

        // NB: do NOT build the pipeline here. The NPU probe loads a 644 MB HTP
        // context cold (multi-second) and onStartCommand runs on the main thread —
        // doing it here would ANR. The worker thread builds the pipeline (probe and
        // all) before it starts draining windows.
        alive = true
        running = true
        lastState = null           // fresh run starts with no reading
        startCaptureAndWorker()
        Log.i("Slashh", "StressMonitorService started (background NPU monitor)")
        return START_STICKY
    }

    private fun startCaptureAndWorker() {
        // Capture stays cheap: publish the newest window for scoring, and feed the same
        // window to the Whisper accumulator (it keeps only the newest hop). Both are O(hop).
        capture = AudioCapture { window ->
            latest.set(window)
            coordinator?.onWindow(window)
            wavlmCoordinator?.onWindow(window)
        }.also {
            try {
                it.start()
            } catch (e: Throwable) {
                Log.e("Slashh", "mic capture failed to start: ${e.message}", e)
                stopSelf()
            }
        }

        worker = Thread({
            // Build the pipeline here, on the worker — the NPU probe is slow (cold
            // 644 MB context load) and must never run on the main thread. Windows
            // captured while we probe just pile up in `latest` (latest-wins) and the
            // newest one is scored as soon as we enter the loop.
            val pipeline = try {
                buildPipeline()
            } catch (e: Throwable) {
                Log.e("Slashh", "Failed to build pipeline: ${e.message}", e)
                // Can't recover from pipeline build failure - stop the service
                alive = false
                stopSelf()
                return@Thread
            }
            
            prefs.enterThreshold?.let { pipeline.enterThreshold = it }
            prefs.releaseThreshold?.let { pipeline.releaseThreshold = it }

            // Bring up Whisper transcription + the text-stress / fusion models. Best-effort:
            // if the assets or the whisper helper are missing, the meter stays audio-only.
            try { attachTextFusion(pipeline) } catch (e: Throwable) {
                Log.w("Slashh", "text/fusion attach failed (audio-only): ${e.message}", e)
            }

            while (alive) {
                val window = latest.getAndSet(null)
                if (window == null) {
                    try { Thread.sleep(50) } catch (_: InterruptedException) { break }
                    continue
                }
                try {
                    val state = pipeline.onWindow(window)
                    lastState = state          // publish for the in-app meter to mirror
                    coordinator?.let {
                        lastTranscript = it.transcript
                        whisperBackend = if (it.backend == Transcriber.Backend.QNN_NPU)
                            "Snapdragon NPU (Whisper)" else "—"
                    }
                    lastTextScore = state.textScore
                    val now = System.currentTimeMillis()
                    calmCue.onState(state.stressed, now)
                    if (calmCue.justTriggered) fireStressNotification()
                    Log.d("Slashh", "monitor voiced=${state.voiced} raw=${state.rawScore} level=${state.level} stressed=${state.stressed}")
                    Log.d("Slashh", "FUSE audio=${state.audioStress} text=${state.textScore} fused=${state.fused}")
                } catch (e: Throwable) {
                    Log.w("Slashh", "monitor scoring error: ${e.message}", e)
                    // On native crash or scoring failure, sleep longer to avoid tight loop
                    try { Thread.sleep(1000) } catch (_: InterruptedException) { break }
                }
            }
        }, "slashh-monitor").apply { isDaemon = true; start() }
    }

    /** Pick the next relief from the user's prefs and post the heads-up notification. */
    private fun fireStressNotification() {
        val enabled = prefs.enabled()
        if (enabled.isEmpty()) return
        val (type, idx) = ReliefSelector.next(enabled, prefs.lastIndex)
        prefs.lastIndex = idx
        if (prefs.notify) ReliefNotifier.notify(this, type)
        Log.i("Slashh", "STRESS DETECTED (NPU) -> notified: ${type.key}")
    }

    /**
     * Same scorer selection as the in-app pipeline, but the WavLM path runs on the
     * NPU **out of process** through [NpuHelperScorer] (the in-app QNN delegate is
     * SELinux-blocked on a retail S25). The channel is the app's external files dir,
     * shared read-write with the shell-domain `npu_helper.sh` via group ext_data_rw.
     *
     * We pick the NPU path by *probing the live helper*, not by a sentinel file: a
     * .pte staged in the app dir is the wrong signal (the helper runs forward_2.bin
     * from its own rig, not an in-app .pte) and the .pte may not be present at all.
     * The probe sends one synthetic window and waits up to ~15 s for the helper to
     * answer — which both proves the channel works AND warms the HTP context so the
     * first real window is fast. If the helper doesn't answer we wire the CPU
     * StressNet directly, so we never pay a per-window NPU timeout on every hop.
     *
     * MUST run on the worker thread (the probe's cold context load is multi-second).
     */
    private fun buildPipeline(): StressPipeline {
        val channelDir = getExternalFilesDir(null)
        // WavLM on the Hexagon NPU reloads a 614 MB context per forward. If free memory is low,
        // even the probe's single load can push the device past its limit and the
        // lowmemorykiller reaps the app. Only engage WavLM when there's real headroom; otherwise
        // run the responsive energy + text-fusion path (Whisper still runs — its contexts are
        // small). This self-adapts: with memory free (e.g. after a reboot) WavLM engages.
        val memMb = memAvailableKb() / 1024
        val wavlmHasHeadroom = memMb >= 1_500
        if (channelDir != null && !wavlmHasHeadroom) {
            Log.w("Slashh", "monitor: skipping WavLM NPU — low memory (${memMb} MB free, need ~1500); energy + text fusion")
        }
        if (channelDir != null && wavlmHasHeadroom) {
            // Score WavLM ASYNC via [WavLmCoordinator] (never blocking the meter) with a long
            // per-window timeout and a probe of up to 30 s (the first request warms the cold
            // context). fallback=null so a miss reads 0 and the energy signal carries the meter —
            // NOT the in-app CPU StressNet, whose native forward SIGSEGVs here.
            val scorer = NpuHelperScorer(channelDir, fallback = null, timeoutMs = 12_000L)
            if (scorer.probe(30_000L)) {
                Log.i("Slashh", "monitor: WavLM on NPU (probe OK, channel ${channelDir.path})")
                backend = "Snapdragon NPU (WavLM) + text fusion"
                val coord = WavLmCoordinator(scorer)
                wavlmCoordinator = coord
                coord.start()
                // Energy drives the responsive meter; WavLM is polled async as the fusion's
                // audio leg, so both NPU models (WavLM + Whisper) run without freezing the UI.
                val pipe = StressPipeline(StressClassifier { 0f })
                pipe.audioModelProvider = { coord.score() }
                return pipe
            }
            Log.w("Slashh", "monitor: NPU WavLM did not answer probe; using energy signal")
        }
        // No NPU helper: energy-only audio (the verified-reliable signal), still feeds fusion.
        backend = "On-device (energy + text fusion)"
        return StressPipeline(StressClassifier { 0f })
    }

    /**
     * Best-effort CPU StressNet as a raw-waveform scorer: log-mel features → tiny
     * StressNet. Returns null if no CPU model ships in assets, so the monitor
     * degrades to "no stress" rather than crashing. Used as the NPU-helper fallback.
     */
    private fun cpuRawScorerOrNull(): RawWaveScorer? {
        return try {
            val name = "stress_model.pte"
            val present = runCatching { assets.list("")?.toSet() ?: emptySet() }.getOrDefault(emptySet())
            if (name !in present) {
                Log.w("Slashh", "CPU fallback: $name not found in assets")
                null
            } else {
                val outFile = File(filesDir, name)
                if (!outFile.exists()) {
                    assets.open(name).use { input -> outFile.outputStream().use { input.copyTo(it) } }
                }
                // Validate file exists and has reasonable size before loading
                if (!outFile.exists() || outFile.length() < 1000) {
                    Log.w("Slashh", "CPU fallback: model file invalid (size=${outFile.length()})")
                    null
                } else {
                    val cpu = ExecuTorchStressClassifier(outFile.absolutePath)
                    cpuFallback = cpu
                    val logMel = LogMel()
                    RawWaveScorer { pcm -> cpu.score(logMel.extractFlat(pcm)) }
                }
            }
        } catch (e: Throwable) {
            Log.w("Slashh", "CPU fallback unavailable: ${e.message}", e)
            null
        }
    }

    /**
     * Bring up the second feature: Whisper transcription (out-of-process on the NPU) + the
     * text-stress classifier, fused with the audio model's per-window score for a more
     * confident reading. All best-effort — a missing asset or a down whisper helper leaves the
     * meter audio-only (honest degradation, never a crash). Runs on the worker thread.
     */
    private fun attachTextFusion(pipeline: StressPipeline) {
        val text = loadTextClassifierOrNull()
        val fusion = loadFusionScorerOrNull()
        textClassifier = text
        fusionScorer = fusion

        // Build the text scorer via ?.let so there's no smart-cast / if-branch-lambda ambiguity
        // (an `if (x != null) { t -> ... }` branch miscompiles to a Unit-returning block).
        val textScorerFn: ((String) -> Float?)? = text?.let { c -> { s: String -> c.score(s) } }
        val coord = TranscriptionCoordinator(
            transcriber = NpuWhisperTranscriber(this),
            textScorer = textScorerFn,
        )
        coordinator = coord
        coord.start()

        // Always expose the text score to the UI; only FUSE when the fusion model is present.
        pipeline.textScoreProvider = { coord.textScore() }
        fusion?.let { f -> pipeline.fuse = { a, t -> f.fuse(a, t) } }
        Log.i(
            "Slashh",
            if (fusion != null) "monitor: text+fusion ON (whisper coordinator + fusion.pte loaded)"
            else "monitor: transcription ON but fusion.pte absent — meter stays audio-only",
        )

        // One-shot on-device sanity check: prove the text model discriminates and the fusion
        // lifts a calm-energy reading when the words are stressed. Evidence in logcat only.
        text?.let { c ->
            val sStress = c.score("i am so stressed and overwhelmed i cannot cope with this")
            val sCalm = c.score("the weather is calm and pleasant we relaxed in the garden")
            Log.i("Slashh", "text-stress check: stressed-sentence->$sStress  calm-sentence->$sCalm")
            val f = fusion
            if (f != null && sStress != null && sCalm != null) {
                Log.i(
                    "Slashh",
                    "fusion check @ audio=0.30: +stressedText->${f.fuse(0.30f, sStress)}  " +
                        "+calmText->${f.fuse(0.30f, sCalm)}  (audio-only stays ~0.30)",
                )
            }
        }
    }

    private fun loadTextClassifierOrNull(): TextStressClassifier? = try {
        val pte = copyAssetToFiles("text_stress.pte")
        val vocab = assets.open("text_stress_vocab.txt").bufferedReader(Charsets.UTF_8).useLines { seq ->
            seq.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
        if (pte == null || vocab.isEmpty()) null
        else TextStressClassifier(pte, vocab).also {
            it.score("everything is fine")     // warm + validate the native forward
            Log.i("Slashh", "text classifier loaded (vocab=${vocab.size})")
        }
    } catch (e: Throwable) {
        Log.w("Slashh", "text classifier unavailable: ${e.message}", e); null
    }

    private fun loadFusionScorerOrNull(): FusionScorer? = try {
        val pte = copyAssetToFiles("fusion.pte")
        if (pte == null) null else FusionScorer(pte).also {
            it.fuse(0.5f, 0.5f)                // warm + validate the native forward
            Log.i("Slashh", "fusion scorer loaded")
        }
    } catch (e: Throwable) {
        Log.w("Slashh", "fusion scorer unavailable: ${e.message}", e); null
    }

    /** Copy an asset to filesDir (ExecuTorch Module loads from a real path). Null if absent. */
    private fun copyAssetToFiles(name: String): String? = try {
        val out = File(filesDir, name)
        if (!out.exists() || out.length() < 100) {
            assets.open(name).use { input -> out.outputStream().use { input.copyTo(it) } }
        }
        if (out.exists() && out.length() > 100) out.absolutePath else null
    } catch (e: Throwable) {
        Log.w("Slashh", "asset $name not copied: ${e.message}"); null
    }

    /** Free memory in kB from /proc/meminfo (MemAvailable), or 0 if unreadable. */
    private fun memAvailableKb(): Long = try {
        java.io.File("/proc/meminfo").useLines { lines ->
            lines.firstOrNull { it.startsWith("MemAvailable") }
                ?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull() ?: 0L
        }
    } catch (_: Throwable) { 0L }

    private fun startForegroundCompat() {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val n: Notification = NotificationCompat.Builder(this, CHANNEL_MONITOR)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle("Slashh is listening")
            .setContentText("Monitoring for stress on-device — nothing leaves your phone")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setContentIntent(open)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(MONITOR_NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(MONITOR_NOTIF_ID, n)
        }
    }

    private fun ensureMonitorChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(NotificationManager::class.java)
            if (mgr.getNotificationChannel(CHANNEL_MONITOR) == null) {
                val ch = NotificationChannel(
                    CHANNEL_MONITOR, "Background monitor", NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Persistent notice that on-device stress monitoring is active"
                    setShowBadge(false)
                }
                mgr.createNotificationChannel(ch)
            }
        }
    }

    override fun onDestroy() {
        alive = false
        running = false
        lastState = null
        backend = "stopped"
        worker?.interrupt()
        worker = null
        capture?.stop()
        capture = null
        cpuFallback?.close()
        cpuFallback = null
        coordinator?.stop()
        coordinator = null
        wavlmCoordinator?.stop()
        wavlmCoordinator = null
        runCatching { textClassifier?.close() }; textClassifier = null
        runCatching { fusionScorer?.close() }; fusionScorer = null
        lastTranscript = ""
        lastTextScore = null
        whisperBackend = "—"
        Log.i("Slashh", "StressMonitorService stopped")
        super.onDestroy()
    }

    companion object {
        const val CHANNEL_MONITOR = "slashh_monitor"
        const val MONITOR_NOTIF_ID = 1002
        const val ACTION_STOP = "ai.slashh.action.STOP_MONITOR"

        /** Reflects whether the service is currently running (for the UI toggle). */
        @Volatile
        var running = false
            private set

        /**
         * The most recent scored state, published so the in-app meter can mirror the
         * live NPU reading while the monitor owns the mic — otherwise the gauge would
         * sit on its idle "Listening…" dash the whole time the service is active.
         * Null until the first window is scored, and again after the service stops.
         * The monitor and the gauge stay independent: the service decides when to
         * *notify*; this only lets an open app reflect the same numbers.
         */
        @Volatile
        var lastState: StressPipeline.StressState? = null
            private set

        /** Which scorer the monitor settled on (NPU/WavLM vs CPU), for the UI badge. */
        @Volatile
        var backend: String = "starting"
            private set

        /** Most recent Whisper transcript (the second feature), for the UI / honesty panel. */
        @Volatile
        var lastTranscript: String = ""
            private set

        /** Latest fused-in text-stress score in [0,1], or null when no fresh transcript. */
        @Volatile
        var lastTextScore: Float? = null
            private set

        /** Whisper backend badge ("Snapdragon NPU (Whisper)" when the helper is serving). */
        @Volatile
        var whisperBackend: String = "—"
            private set

        fun start(context: Context) {
            val i = Intent(context, StressMonitorService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(i)
            } else {
                context.startService(i)
            }
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, StressMonitorService::class.java).setAction(ACTION_STOP)
            )
        }
    }
}
