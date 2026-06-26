package ai.slashh

import ai.slashh.audio.AudioCapture
import ai.slashh.audio.ExecuTorchStressClassifier
import ai.slashh.audio.StressPipeline
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

/**
 * Minimal end-to-end wiring (M6 + M7 host): permission → copy `.pte` out of
 * assets → capture → pipeline → on-screen level. The polished stress meter (M8)
 * and calming intervention (M9) build on top of this loop.
 *
 * Everything runs on-device; there is no network code anywhere in this path.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var levelView: TextView
    private var capture: AudioCapture? = null
    private var classifier: ExecuTorchStressClassifier? = null
    private lateinit var pipeline: StressPipeline

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startListening() else levelView.text = getString(R.string.need_mic) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        levelView = TextView(this).apply { textSize = 22f; text = "…" }
        setContentView(levelView)

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
    }

    private fun startListening() {
        if (capture != null) return
        capture = AudioCapture { window ->
            val state = pipeline.onWindow(window)
            val text = when {
                !state.voiced && state.level == null -> "Listening…"
                state.level == null -> "Listening…"
                else -> "Stress: %.0f%%%s".format(
                    state.level!! * 100,
                    if (state.stressed) "  ⚠ take a breath" else "",
                )
            }
            runOnUiThread { levelView.text = text }
            Log.d("Slashh", "voiced=${state.voiced} raw=${state.rawScore} level=${state.level} stressed=${state.stressed}")
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
