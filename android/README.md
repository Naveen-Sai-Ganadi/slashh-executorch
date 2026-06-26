# Slashh — Android app

On-device voice-stress detection. Mic → VAD → log-mel → ExecuTorch `.pte` →
stress meter → calming cue. **No network**: the manifest declares no `INTERNET`
permission, so the whole loop runs in airplane mode (plan §5).

## Module map (`app/src/main/java/ai/slashh/`)

| File | Role | Backlog |
|------|------|---------|
| `audio/AudioConfig.kt` | constants mirrored from `model/audio_config.py` | — |
| `audio/AudioCapture.kt` | 16 kHz mono mic capture → 3 s / 1 s rolling windows | M6 |
| `audio/Vad.kt` | energy + ZCR voice-activity gate (energy lever) | M6 |
| `audio/RealDft.kt` | exact 400-pt real DFT (NumPy `rfft` twin) | M6 |
| `audio/LogMel.kt` | log-mel extractor, parity with `model/features.py` | M6 |
| `audio/StressClassifier.kt` | classifier interface | M7 |
| `audio/ExecuTorchStressClassifier.kt` | ExecuTorch runtime wrapper | M7 |
| `audio/StressPipeline.kt` | VAD gate → features → score → EMA → hysteresis | M6/M8 |
| `MainActivity.kt` | minimal wiring (permission → capture → on-screen level) | M6/M7 |

## The parity gate (M6)

`audio/LogMel.kt` must reproduce the torchaudio features the model was trained
on. The JVM test `LogMelParityTest` loads golden vectors emitted by
`python -m model.golden` (`app/src/test/resources/golden_logmel.json`) and
asserts **max-abs-error < 1e-3** — verified at ~1e-5 on representative audio.
Regenerate the golden file whenever the feature pipeline or `audio_config`
changes.

```bash
# from repo root: refresh golden vectors after any feature change
python -m model.golden
```

## Build & run

Open `android/` in **Android Studio** (it provisions Gradle and the wrapper),
or from the CLI once a JDK 17 + Gradle are installed:

```bash
cd android
gradle wrapper            # first time only — generates ./gradlew
./gradlew testDebugUnitTest    # log-mel parity + VAD + pipeline tests (no device)
./gradlew assembleDebug        # build the APK
./gradlew installDebug         # install to a connected S25 Ultra
# or: adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## ExecuTorch runtime dependency

`app/build.gradle.kts` pulls `org.pytorch:executorch-android`. Align its version
with the pip `executorch` that produced the `.pte` (1.2.0) and bundle the **QNN
backend** libraries to run on the Hexagon NPU. If a matching Maven artifact
isn't available, build the AAR from the ExecuTorch repo (`extension/android`)
and drop it in `app/libs/`.

The model is bundled at `app/src/main/assets/stress_model.pte` (copied from the
repo-root `assets/` after `python -m model.train --export …`). The XNNPACK build
is the portable default; the AI Hub QNN/NPU `.pte` drops in with no Kotlin change.
