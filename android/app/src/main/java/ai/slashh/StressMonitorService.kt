package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.LogMel
import ai.slashh.audio.NpuHelperScorer
import ai.slashh.audio.RawWaveScorer
import ai.slashh.audio.StressClassifier
import ai.slashh.audio.StressPipeline
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
        // Capture stays cheap: just publish the newest window.
        capture = AudioCapture { window -> latest.set(window) }.also {
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

            while (alive) {
                val window = latest.getAndSet(null)
                if (window == null) {
                    try { Thread.sleep(50) } catch (_: InterruptedException) { break }
                    continue
                }
                try {
                    val state = pipeline.onWindow(window)
                    lastState = state          // publish for the in-app meter to mirror
                    val now = System.currentTimeMillis()
                    calmCue.onState(state.stressed, now)
                    if (calmCue.justTriggered) fireStressNotification()
                    Log.d("Slashh", "monitor voiced=${state.voiced} raw=${state.rawScore} level=${state.level} stressed=${state.stressed}")
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
        if (channelDir != null) {
            val scorer = NpuHelperScorer(channelDir, fallback = cpuRawScorerOrNull())
            if (scorer.probe()) {
                Log.i("Slashh", "monitor scorer: WavLM via NPU helper (probe OK, channel ${channelDir.path})")
                backend = "Snapdragon NPU (WavLM)"
                return StressPipeline(StressClassifier { 0f }, rawScorer = scorer)
            }
            Log.w("Slashh", "monitor scorer: NPU helper did not answer probe; using CPU fallback")
        }
        // NPU helper not live — fall back to the in-process CPU StressNet entirely.
        val cpu = cpuRawScorerOrNull()
        if (cpu != null) {
            Log.i("Slashh", "monitor scorer: CPU StressNet (NPU helper unavailable)")
            backend = "CPU (StressNet)"
            return StressPipeline(StressClassifier { 0f }, rawScorer = cpu)
        }
        Log.w("Slashh", "monitor scorer: none loadable; monitor will report no stress")
        backend = "no model"
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
