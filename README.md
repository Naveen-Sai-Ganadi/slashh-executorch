# Slashh — On-device Voice Stress Detection

**Slashh** notices when your voice is under stress and offers a calming nudge — **entirely on
your phone**, in real time, with **no network, no recording, and no LLM**. It runs **two neural
models on the Snapdragon Hexagon NPU** and fuses them into one confident reading:

- 🎙️ **WavLM** — an audio stress model that listens to *how* you speak.
- 📝 **Whisper-Tiny** — transcribes *what* you say; a trained text classifier scores the words.
- 🔗 A small **fusion perceptron** combines the two into a single stress probability.

Built for the **[Qualcomm × Meta ExecuTorch Hackathon](https://lablab.ai/ai-hackathons/qualcomm-x-meta-executorch-hackathon)** (San Francisco, June 27–28 2026) on a
**Samsung Galaxy S25 Ultra** (Snapdragon 8 Elite). Everything in the inference hot path runs
locally; the app declares **no `INTERNET` permission** and works in airplane mode.

> **Want the full story, deep-dive architecture, NPU runtime details, model training, and the
> research log of everything we tried?** → **[`docs/`](docs/index.md)**

![Architecture](docs/stress-architecture.svg)

---

## Table of contents

1. [What it does](#what-it-does)
2. [How it works (current architecture)](#how-it-works-current-architecture)
3. [Team](#team)
4. [Prerequisites](#prerequisites)
5. [Part A — Build & install the app](#part-a--build--install-the-app)
6. [Part B — Bring the two models onto the NPU](#part-b--bring-the-two-models-onto-the-npu)
7. [Verify it's working](#verify-its-working)
8. [Troubleshooting & debugging](#troubleshooting--debugging)
9. [Thresholds & tuning](#thresholds--tuning)
10. [How this maps to the judging criteria](#how-this-maps-to-the-judging-criteria)
11. [Repository layout](#repository-layout)
12. [License](#license)

---

## What it does

A foreground service listens to the microphone, scores **one 3-second window every second**, and
raises a gentle **relief nudge** (local notification + an in-app breathing/sound/colour overlay)
when stress *sustains* — not on a single spike. The on-screen meter and a **Settings → Live
transcription** panel show the live transcript and the three signals (text / audio / fused) so you
can see exactly why it reacts.

Reframed as **voice tension / arousal** detection, it supports relaxation and reflection. It is
**not a medical device**.

## How it works (current architecture)

```
mic 16 kHz ─▶ AudioCapture (3 s window / 1 s hop)
                 │
     ┌───────────┴────────────┐
     ▼                        ▼
  AUDIO score              TEXT score
  VAD gate                 WhisperBuffer (10 s) ─▶ Whisper-Tiny ASR  [NPU]
  WavLM audio model [NPU]      ─▶ transcript ─▶ text classifier (BoW 2000 → MLP)
  (energy fallback)            ─▶ text score ∈ [0,1]
  ─▶ audio score ∈ [0,1]
     └───────────┬────────────┘
                 ▼
   FUSION  Linear(2,1)+sigmoid (symmetric)   fires when  audio + text ≥ 0.9
                 ▼  fused stress ∈ [0,1]
   EMA (α=0.4) ─▶ hysteresis latch (enter 0.62 / release 0.45)
                 ▼
            STRESSED ─▶ relief nudge
```

Both neural models execute on the **Hexagon NPU out-of-process** (a retail S25 SELinux-blocks the
app from the cDSP, so a shell-domain helper drives the QNN context binaries and exchanges data
over a file channel in the app's own external-files dir — no extra permission, audio never leaves
the device). The full picture, every constant, and every threshold are in
**[`docs/architecture.md`](docs/architecture.md)** and the diagram above
([`docs/stress-architecture.svg`](docs/stress-architecture.svg) / `.png`).

## Team

| Member | Email | GitHub |
|--------|-------|--------|
| Suma Katabattuni | sumakbn@gmail.com | [@katabattunisuma](https://github.com/katabattunisuma) |
| Naveen Sai Ganadi | naveenganadi@gmail.com | [@Naveen-Sai-Ganadi](https://github.com/Naveen-Sai-Ganadi) |
| Rohit Somisetty | rohit.somisetty@gmail.com | [@Rohit-Somisetty](https://github.com/Rohit-Somisetty) |

---

## Prerequisites

**Hardware**
- A **Snapdragon 8 Elite** Android phone — built & verified on a **Samsung Galaxy S25 / S25 Ultra**
  (`SM8750`, Android 15 / One UI 7). USB debugging enabled.
- The S25 Ultra uses **16 KB memory pages** — any native libs must be 16 KB-aligned (the shipped
  ones are).

**Build host** (we used **macOS**)
- **Android Studio** (provides the bundled JBR used as `JAVA_HOME`) + **Android SDK** with
  `platform-tools` (`adb`). Default `adb` path used below: `~/Library/Android/sdk/platform-tools/adb`.
- **Gradle 8.9 / AGP 8.5.2** via the committed wrapper (`android/gradlew`), **JDK 17+** (the
  Android Studio JBR is OpenJDK 21).
- **Node 22 + npm 10** (for the React/Vite web UI bundle).
- **Python 3.11** (only needed to retrain/re-export the models — the shipped `.pte`s are committed).
- *(Optional, for the NPU daemons)* the **Qualcomm AI Runtime (QAIRT/QNN) SDK** — **2.45** for the
  Whisper rig and **2.37** for the WavLM rig. See [`docs/npu-runtime.md`](docs/npu-runtime.md).

> The app **runs without the NPU daemons** — it falls back to the responsive vocal-energy signal
> (and to text fusion as soon as the Whisper daemon is up). Part B turns on the full
> two-models-on-NPU experience.

Set these once in your shell (adjust paths to your machine):

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"
export ADB="$ANDROID_HOME/platform-tools/adb"
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
```

---

## Part A — Build & install the app

### 1. Clone

```bash
git clone https://github.com/Naveen-Sai-Ganadi/slashh-executorch.git
cd slashh-executorch
```

### 2. Build the web UI bundle (React + Vite → Android assets)

The UI is a React/Vite app embedded in a WebView; `vite build` emits into the Android assets dir.

```bash
npm install                # installs Vite + deps (overlaid lockfile → use install, not ci)
npm run build              # → android/app/src/main/assets/www/
```

### 3. Build the debug APK

```bash
cd android
./gradlew :app:assembleDebug          # uses $JAVA_HOME (Android Studio JBR)
# → app/build/outputs/apk/debug/app-debug.apk
cd ..
```

You can also run the (offline, no-device) unit tests — log-mel parity, VAD, pipeline, and the
text-feature/Python↔Kotlin parity test:

```bash
cd android && ./gradlew :app:testDebugUnitTest && cd ..
```

### 4. Install on the phone

```bash
$ADB install -r android/app/build/outputs/apk/debug/app-debug.apk
```

> **Signature mismatch?** If a differently-signed `ai.slashh` build is already on the device, the
> install fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`. Uninstall first (this wipes that app's
> local data), then install:
> ```bash
> $ADB uninstall ai.slashh
> $ADB install android/app/build/outputs/apk/debug/app-debug.apk
> ```

### 5. Grant permissions & launch

```bash
$ADB shell pm grant ai.slashh android.permission.RECORD_AUDIO
$ADB shell pm grant ai.slashh android.permission.POST_NOTIFICATIONS
$ADB shell am start -n ai.slashh/.MainActivity
```

(Or just open the app and tap through the permission prompts. On the login screen, **"Dev skip"**
bypasses auth for testing — all account state is local/on-device, SHA-256, no network.)

At this point the app is **live on energy + (once Part B's Whisper daemon is up) text fusion**.

---

## Part B — Bring the two models onto the NPU

Both models run out-of-process via small shell-domain helpers. **The large QNN context binaries
are not vendored in the repo** (≈ 0.6–0.7 GB total); how to obtain/build them is in
**[`docs/npu-runtime.md`](docs/npu-runtime.md)**. Once they're staged on the device:

### 1. Whisper-Tiny ASR daemon (text path)

`tools/whisper/` ships the runner binary, tokenizer, mel filterbank, and helper. It stages the rig
to `/data/local/tmp/whisper_rig`, points it at the app's channel
(`/sdcard/Android/data/ai.slashh/files`), and starts a resident `--watch` daemon **plus** a small
relay that keeps the transcripts app-readable across the FUSE mount:

```bash
ADB="$ADB" ./tools/whisper/npu_helper_whisper.sh start
# → STATUS: RUNNING ; relay: running
./tools/whisper/npu_helper_whisper.sh status     # check it later
```

### 2. WavLM audio-stress daemon (audio path)

The WavLM rig lives at `/data/local/tmp/qnntest` (`forward_2.bin` 614 MB + `qnn-net-run` + QAIRT
2.37 libs). Start it (it watches the same channel with `npu_*` markers — they never collide with
Whisper's `whisper_*` markers):

```bash
$ADB shell "cd /data/local/tmp/qnntest && nohup sh npu_helper.sh > npu_helper.log 2>&1 < /dev/null &"
```

> **Memory note:** WavLM's runner reloads a 614 MB context **per call**. On a fresh boot the S25 has
> ~5–7 GB free and both daemons coexist fine; if free memory is low the app's **memory guard**
> (`≥ 1500 MB`) skips WavLM and runs energy + text fusion instead. Both daemons die on reboot and
> must be restarted. Full rationale in [`docs/npu-runtime.md`](docs/npu-runtime.md).

### 3. Relaunch the app

```bash
$ADB shell "am force-stop ai.slashh; am start -n ai.slashh/.MainActivity"
```

On launch the monitor probes the WavLM helper (up to 30 s) and, if it answers, routes the audio
leg through **WavLM on the NPU**; Whisper transcribes on the NPU in parallel.

---

## Verify it's working

```bash
# 1. Both NPU daemons alive?
$ADB shell "ps -A -o ARGS | grep -q 'whisper_qnn --watch' && echo whisper:UP || echo whisper:DOWN"
$ADB shell "pgrep -f npu_helper.sh >/dev/null && echo wavlm:UP || echo wavlm:DOWN"

# 2. Watch the live pipeline (speak into the phone):
$ADB logcat -d | grep ' Slashh' | grep -E 'WavLM on NPU|wavlm\[NPU\]|whisper\[|FUSE audio'
```

Expected, live:
- `monitor: WavLM on NPU (probe OK ...)` and `wavlm[NPU] raw=… smoothed=…` every ~2 s.
- `whisper[QNN_NPU] -> "…your words…" textScore=…` every ~4 s.
- `FUSE audio=… text=… fused=…` per window.

In the app, open **Settings → Live transcription** to see the transcript and the **text / audio /
fused** percentages update as you speak. Say a clearly **anxious** sentence (*"I'm so overwhelmed,
I can't cope"*) — text and audio both rise, the fused score climbs past the latch, and a relief
nudge fires. Ordinary chatter stays calm.

> **No NPU binaries? You can still run and see results.** Part A alone gives you a working app:
> the meter moves on the **vocal-energy** signal and relief nudges fire. Text + audio fusion light
> up once Part B's daemons are running. So the *minimum* path to "install → speak → see the meter
> react → get a relief nudge" is just **Part A**.

---

## Troubleshooting & debugging

If something doesn't work, you almost never have to guess — the app logs every stage to `logcat`
under the tags `Slashh` (pipeline/monitor), `SlashhVAD` (per-window voice activity), and
`SlashhWebView` (the WebView's JS console). Start with the diagnostics toolkit, then jump to the
symptom that matches.

### Diagnostics toolkit (copy/paste)

```bash
# Is the phone connected & authorized?
$ADB devices -l                         # must list your device as "device" (not "unauthorized")

# Is the app running & focused?
$ADB shell ps -A | grep ai.slashh
$ADB shell dumpsys window | grep mCurrentFocus

# Live pipeline (speak into the phone while this runs):
$ADB logcat -d | grep ' Slashh' | grep -E 'wavlm\[NPU\]|whisper\[|FUSE audio|monitor:'

# The on-device file channel both NPU daemons talk through:
$ADB shell run-as ai.slashh ls -la files/                       # cached .pte models (internal)
$ADB shell ls -la /sdcard/Android/data/ai.slashh/files/         # npu_*/whisper_* handshake markers

# The two NPU daemon logs:
./tools/whisper/npu_helper_whisper.sh log                       # Whisper daemon
$ADB shell "tail -20 /data/local/tmp/qnntest/npu_helper.log"    # WavLM daemon

# What's actually on screen right now (saves a PNG you can open):
$ADB exec-out screencap -p > /tmp/slashh_screen.png
```

### `adb: command not found` / device not listed

- Use the full path: `export ADB="$HOME/Library/Android/sdk/platform-tools/adb"` (install
  *platform-tools* via Android Studio's SDK Manager if missing).
- Device shows as `unauthorized` → unlock the phone and accept the **"Allow USB debugging?"**
  prompt. Still stuck? `$ADB kill-server && $ADB start-server`, replug the cable, re-accept.
- Nothing at all → enable **Developer options → USB debugging** on the phone, and use a *data*
  USB cable (not charge-only).

### Build / install problems

- **`npm run build` not run first** → the WebView shows a blank/old screen. The UI is bundled into
  `android/app/src/main/assets/www/` by Vite, so always `npm run build` *before* `assembleDebug`.
- **`JAVA_HOME` / Gradle errors** → point `JAVA_HOME` at the Android Studio JBR (see Prerequisites);
  the build needs JDK 17+.
- **`INSTALL_FAILED_UPDATE_INCOMPATIBLE` (signature mismatch)** → a differently-signed `ai.slashh`
  is already installed. `$ADB uninstall ai.slashh` then install again (this wipes that app's local
  data, including any cached models — see the "model update doesn't take effect" note below).

### App opens but the meter is frozen at `0%` / says "resting"

1. **Mic permission** — the most common cause. Grant it and relaunch:
   ```bash
   $ADB shell pm grant ai.slashh android.permission.RECORD_AUDIO
   $ADB shell "am force-stop ai.slashh; am start -n ai.slashh/.MainActivity"
   ```
2. Confirm the foreground monitor is alive: `logcat … | grep 'SlashhVAD'` should print a
   `voiced=… rms=…` line roughly every second. No lines → the service isn't capturing; re-grant the
   mic and relaunch.
3. Meter moves but stays *very* low even when you speak up → it's the **energy anchors**. Open
   **Settings → Calibrate** and follow the calm/stressed prompts, or nudge **Settings → Stress
   sensitivity**.

### Transcript / `text` / `fused` is always `null` (no text fusion)

This is usually **expected behaviour**, not a bug — work through it in order:

1. **Is the Whisper daemon up?** `./tools/whisper/npu_helper_whisper.sh status` → expect
   `RUNNING`. If not, `./tools/whisper/npu_helper_whisper.sh start`.
2. **The 20-second freshness window.** A text score is only published while a transcript is fresh
   (`freshnessMs = 20 s`, `TranscriptionCoordinator.kt`). Go quiet for >20 s and `text`/`fused`
   correctly revert to `null` and the meter falls back to **audio-only**. Speak again and they
   return. This is the designed "honest degradation."
3. **The meaningful-speech gate.** On silence/noise Whisper emits placeholders — `[BLANK_AUDIO]`,
   `(music)`, `[INAUDIBLE]` — which are stripped (`isMeaningful()`), so no text score is published.
   Check the daemon log: `./tools/whisper/npu_helper_whisper.sh log` should show
   `samples -> "real words"`, not only bracketed tokens. Speak clear sentences close to the mic.
4. **Wrong channel/package.** The daemon must watch **`/sdcard/Android/data/ai.slashh/files`**. If
   you see another package (e.g. `com.scamshield.app`) being polled, that's a stale/misconfigured
   runner — restart via `npu_helper_whisper.sh` which points at the correct dir.
5. **`fusion.pte` absent** → the log says `transcription ON but fusion.pte absent — meter stays
   audio-only`. The app still shows the text % but won't fuse. Rebuild so the asset is bundled.

### `fusion scorer unavailable` in the logs

The fusion model failed its warm-up forward (`fuse(0.5, 0.5)`) at load — usually a missing or
corrupt `.pte`. Check it's present and non-trivial:

```bash
$ADB shell run-as ai.slashh ls -la files/fusion.pte     # should be a couple of KB, not 0
```

Re-push it (see below) or `$ADB shell pm clear ai.slashh` and relaunch to force a fresh copy from
the APK assets.

### High text stress, but `fused` stays low

The fusion is a symmetric perceptron with a **sum** decision boundary
(`DECISION_BOUNDARY_SUM`, `model/fusion.py`): `fused` crosses 0.5 when `audio + text` clears it.
The live WavLM audio leg sits low (~0.27, rarely > 0.5), so if the boundary is too high a strong
text score gets crushed. It ships at **0.9**. To make it more (or less) responsive, change
`DECISION_BOUNDARY_SUM` and re-export + redeploy:

```bash
# 1. edit DECISION_BOUNDARY_SUM in model/fusion.py, then re-export both artifacts:
.venv/bin/python - <<'PY'
import torch
from model.fusion import load_fusion_model, export_to_pte, CHECKPOINT_PATH, PTE_PATH, DECISION_BOUNDARY_SUM
m = load_fusion_model()                                   # keep the trained (seeded) weights
with torch.no_grad():
    m.net.bias.fill_(-(DECISION_BOUNDARY_SUM / 2.0) * float(m.net.weight.sum()))
torch.save({"model": m.state_dict()}, CHECKPOINT_PATH)
PTE_PATH.write_bytes(export_to_pte(m))
print("re-exported fusion.pte at boundary", DECISION_BOUNDARY_SUM)
PY

# 2. hot-swap onto the device WITHOUT reinstalling (see the gotcha below), then relaunch:
$ADB push android/app/src/main/assets/fusion.pte /data/local/tmp/fusion_new.pte
$ADB shell "run-as ai.slashh sh -c 'cat /data/local/tmp/fusion_new.pte > files/fusion.pte'"
$ADB shell "am force-stop ai.slashh; am start -n ai.slashh/.MainActivity"
```

On relaunch the service prints a self-check you can read back:
`fusion check @ audio=0.30: +stressedText->0.95 +calmText->0.05` — a high text score should now
produce a high fused value at low audio.

### Updating a bundled model doesn't take effect

**Gotcha:** the app copies each `.pte` from the APK assets into its private `files/` dir **once**
and never overwrites it (`copyAssetToFiles` only copies when the file is absent or < 100 bytes).
So a plain `install -r` with a new asset **won't** replace the cached model. Either:

- **Hot-swap** the file in `files/` directly (the `run-as … cat > files/…` recipe above), or
- `$ADB shell pm clear ai.slashh` to wipe the cache (then re-grant permissions), or
- `$ADB uninstall ai.slashh` and reinstall fresh.

### WavLM isn't running on the NPU (audio stays energy-only)

`logcat … | grep 'monitor:'` tells you which branch the audio leg took:

- `monitor: WavLM on NPU (probe OK …)` → good, you're on the NPU.
- `monitor: skipping WavLM NPU — low memory (… MB free, need ~1500)` → the **memory guard**
  tripped. WavLM reloads a 614 MB context per call; free RAM (close apps) or reboot, then restart
  the daemon and the app.
- Neither line / `wavlm[NPU]` never appears → the daemon isn't answering. Check it's alive
  (`$ADB shell pgrep -f npu_helper.sh`) and read `/data/local/tmp/qnntest/npu_helper.log` — a
  `FATAL: missing …` line means the rig is incomplete (see [`docs/npu-runtime.md`](docs/npu-runtime.md)).
- The app probes for up to 30 s at launch; if you started the daemon *after* the app, just relaunch
  the app.

### Daemons disappear after a reboot

Both NPU daemons live in the shell domain and **do not survive a reboot or USB unplug**. Re-run the
two Part B start commands. Quick "are they both up?" check:

```bash
$ADB shell "ps -A -o ARGS | grep -q 'whisper_qnn --watch' && echo whisper:UP || echo whisper:DOWN"
$ADB shell "pgrep -f npu_helper.sh >/dev/null && echo wavlm:UP || echo wavlm:DOWN"
```

### The relief nudge never fires

Stress has to **sustain**, not spike: the latch enters at EMA `0.62` and the score is smoothed
(`α = 0.4`), so a single loud word won't trip it. Raise **Settings → Stress sensitivity** (lowers
the enter threshold), or use **Simulate** on the dashboard to force the relief overlay for a demo.

---

## Thresholds & tuning

Every constant lives in one of a few places (and is listed in the diagram and
[`docs/architecture.md`](docs/architecture.md)):

| What | Value | Where |
|------|-------|-------|
| Audio window / hop | 3 s (48000) / 1 s (16000), 16 kHz | `model/audio_config.py` ↔ `AudioConfig.kt` |
| Energy anchors | calm 0.03 → stress 0.14 | `StressPipeline.kt` / calibration |
| WavLM cadence | throttle 2 s, EMA α=0.5, freshness 30 s | `WavLmCoordinator.kt` |
| WavLM memory guard | engage only if ≥ 1500 MB free | `StressMonitorService.kt` |
| Whisper cadence | 10 s buffer, 4 s, freshness 20 s, timeout 8 s | `TranscriptionCoordinator.kt` / `WhisperConfig.kt` |
| Fusion decision boundary | `audio + text ≥ 0.9` (a strong text score alone can register, since the live WavLM audio leg sits ~0.27 and never carries half of 1.2) | `model/fusion.py` (`DECISION_BOUNDARY_SUM`) |
| Stress latch | enter 0.62 / release 0.45 | `StressPipeline.kt`; in-app **Sensitivity** slider |

The in-app **Settings → Stress sensitivity** slider adjusts the latch live. To re-tune the fusion's
conservativeness, change `DECISION_BOUNDARY_SUM` in `model/fusion.py` and re-export (see
[`docs/models.md`](docs/models.md)).

## How this maps to the judging criteria

| Criterion (weight) | How Slashh addresses it |
|---|---|
| **Technical — NPU utilization, latency, energy (40%)** | **Two** transformer models on the Hexagon NPU (WavLM audio + Whisper-Tiny ASR), out-of-process via QNN context binaries. Tiny ExecuTorch `.pte`s (text classifier, fusion) run on-device. Async coordinators + a memory guard keep both NPU models live without freezing the 1 s meter. |
| **Use-case & innovation (25%)** | Multimodal stress detection that fuses *prosody* (voice) and *semantics* (words) — more confident than either alone — with an always-on, privacy-first relief loop. |
| **Local processing & privacy (15%)** | **No `INTERNET` permission**; audio + transcript never leave the device; no recording, no LLM, no cloud. Runs in airplane mode. |
| **Deployment ease (10%)** | One-command web build + Gradle APK; self-contained helper scripts to bring up the NPU daemons; graceful CPU/energy fallback when the NPU isn't available. |
| **Presentation (10%)** | This README + the [`docs/`](docs/index.md) hub (architecture, NPU runtime, models, research journey) + the architecture diagram. |

## Repository layout

| Path | What |
|------|------|
| `android/app/src/main/java/ai/slashh/` | the Android app (Kotlin): mic → VAD → pipeline → relief |
| `…/audio/` | `StressPipeline`, `WavLmCoordinator`, `NpuHelperScorer`, VAD, log-mel, energy |
| `…/runtime/` | Whisper port: `NpuWhisperTranscriber`, `TranscriptionCoordinator`, `TextStressClassifier`, `FusionScorer`, `TextFeatures` |
| `android/app/src/main/assets/` | `text_stress.pte`, `fusion.pte`, `text_stress_vocab.txt`, `stress_model.pte`, web bundle `www/` |
| `src/` | React/Vite WebView UI (incl. `Settings.tsx` live-transcription panel) |
| `model/` | Python: `text_stress.py`, `fusion.py`, `text_features.py`, `golden_text_fusion.py`, StressNet tooling |
| `tools/whisper/` | the on-device Whisper rig + `npu_helper_whisper.sh` |
| `npu_helper.sh` | the on-device WavLM shell-domain helper |
| `docs/` | architecture, NPU runtime, models, research journey, diagram |

## License

See [LICENSE](LICENSE).
