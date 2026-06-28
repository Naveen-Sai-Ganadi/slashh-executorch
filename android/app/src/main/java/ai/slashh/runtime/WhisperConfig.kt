package ai.slashh.runtime

/**
 * Single source of truth for the on-device Whisper ASR contract, shared by the app-side
 * [NpuWhisperTranscriber] and the shell-domain QNN runner (`npu_helper_whisper.sh` →
 * `whisper_qnn --watch`). If any constant changes, change it in BOTH worlds together — a
 * silent mismatch collapses output.
 *
 * Ported verbatim (markers + protocol) from the proven ScamShield/Edge whisper flow; only
 * the package moved to `ai.slashh.runtime`. Whisper Tiny was exported to QNN context binaries
 * for the S25 Hexagon V79 NPU (see the device rig at `/data/local/tmp/whisper_rig`). The
 * encoder takes a 30 s / 80-mel window; the decoder runs autoregressively. Mel extraction, the
 * decode loop, and detokenization all live in the device-side runner, so the app simply sends
 * raw PCM and reads back UTF-8 text.
 *
 * Channel = the app's own external files dir (`getExternalFilesDir(null)` →
 * `/sdcard/Android/data/ai.slashh/files`) — the SAME file-channel mechanism the proven WavLM
 * stress-model NPU rig already uses ([ai.slashh.audio.NpuHelperScorer]). The whisper markers
 * (`whisper_*`) never collide with the WavLM markers (`npu_*`). No INTERNET, no extra
 * permission; the audio never leaves the device.
 */
object WhisperConfig {
    /** Whisper's fixed input rate; [ai.slashh.audio.AudioConfig.SAMPLE_RATE] already matches. */
    const val SAMPLE_RATE = 16_000

    // ── File channel: the app's external files dir, shared with the shell domain ──────
    const val IN_RAW = "whisper_in.raw"      // app → helper: mono float32 LE PCM window
    const val IN_READY = "whisper_in.ready"  // app → helper: request marker
    const val OUT_TXT = "whisper_out.txt"    // helper → app: UTF-8 transcript of the window
    const val OUT_READY = "whisper_out.ready"
    const val ERR_READY = "whisper_err.ready"

    const val POLL_MS = 50L                  // channel poll interval
    const val TIMEOUT_MS = 8_000L            // abandon a window after this (the decode loop is slow)
}
