package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.StressPipeline
import ai.slashh.ui.BreathOverlayView
import ai.slashh.ui.CalmCue
import ai.slashh.ui.Meter
import ai.slashh.ui.StressMeterView
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.FrameLayout
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

/**
 * End-to-end wiring: permission → copy `.pte` out of assets → capture →
 * pipeline → stress meter (M8) → calming breathing overlay (M9). The overlay is
 * stacked over the meter in a [FrameLayout]; [CalmCue] decides when it appears.
 *
 * Everything runs on-device; there is no network code anywhere in this path.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var meter: StressMeterView
    private lateinit var overlay: BreathOverlayView
    private val calmCue = CalmCue()
    private var capture: AudioCapture? = null
    private var classifier: ExecuTorchStressClassifier? = null
    private lateinit var pipeline: StressPipeline

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startListening() else meter.render(Meter.needMic()) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        meter = StressMeterView(this)
        overlay = BreathOverlayView(this) {
            // user dismissed: hide and stay quiet for the rest of this episode
            calmCue.dismiss(SystemClock.uptimeMillis())
            overlay.hide()
        }.apply { visibility = android.view.View.GONE }
        val root = FrameLayout(this).apply {
            addView(meter)
            addView(overlay)
        }
        setContentView(root)

        val modelPath = copyAsset("stress_model.pte")
        classifier = ExecuTorchStressClassifier(modelPath)
        pipeline = StressPipeline(classifier!!)
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
        overlay.hide()
    }

    private fun startListening() {
        if (capture != null) return
        capture = AudioCapture { window ->
            val state = pipeline.onWindow(window)
            val model = Meter.from(state)
            val showCue = calmCue.onState(state.stressed, SystemClock.uptimeMillis())
            runOnUiThread {
                meter.render(model)
                if (showCue) overlay.show() else overlay.hide()
            }
            Log.d("Slashh", "voiced=${state.voiced} raw=${state.rawScore} level=${state.level} stressed=${state.stressed} cue=$showCue")
        }.also { it.start() }
    }

    /** ExecuTorch loads from a filesystem path; copy the bundled asset out once. */
    private fun copyAsset(name: String): String {
        val outFile = File(filesDir, name)
        if (!outFile.exists()) {
            assets.open(name).use { input ->
                outFile.outputStream().use { input.copyTo(it) }
            }
        }
        return outFile.absolutePath
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
