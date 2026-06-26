# Running the model on the Snapdragon NPU (S25) — runbook

Goal: the live app runs `StressNet` on the **Hexagon NPU** of a Samsung Galaxy
S25 (Snapdragon 8 Elite = `SM8750`, HTP **V79**), via ExecuTorch's QNN backend.

The app is already **NPU-ready**: `MainActivity.loadClassifier()` loads
`stress_model_qnn.pte` if present (else falls back to CPU), and
`app/build.gradle.kts` uses `app/libs/executorch-qnn.aar` if present (else the
Maven CPU AAR). So finishing the NPU path is **drop in 3 artifacts, rebuild**.

## The 3 artifacts and how they're produced

| # | Artifact | Where it goes | Producible on Mac? |
|---|----------|---------------|--------------------|
| 1 | `stress_model_qnn.pte` (QNN-delegated, w8a8) | `app/src/main/assets/` | ✅ via Docker (below) |
| 2 | QNN runtime libs (`libQnnHtp*.so`, `libQnnSystem.so`, V79 skel) | `app/src/main/jniLibs/arm64-v8a/` | ✅ extracted by the same Docker run |
| 3 | QNN-enabled ExecuTorch **AAR** (`libexecutorch.so` built **with** the QNN backend) | `app/libs/executorch-qnn.aar` | ⚠️ heavy build — see below |

### Artifacts 1 + 2 — DONE, one command (no Qualcomm login) ✅
```bash
bash tools/qnn/build_qnn_pte.sh
```
Runs `tools/qnn/export_qnn_pte.py` in a `linux/amd64` container: PT2E-quantizes
`StressNet` (w8a8, calibrated on RAVDESS), lowers it to QNN for `SM8750`, writes
`assets/stress_model_qnn.pte`, and copies the essential HTP-V79 runtime `.so`s
into `jniLibs/arm64-v8a/`. The container mounts the QAIRT Community SDK from
`.qairt/` (run downloads it once; no Qualcomm login) and sets
`LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang` + installs `libc++1`.
The `.pte` is committed; the `.so`s are gitignored (regenerate with this script).

### Artifact 3 — the QNN-enabled AAR (the one hard part)
The Maven `executorch-android` AAR is CPU-only (no QNN). You need `libexecutorch.so`
built with `-DEXECUTORCH_BUILD_QNN=ON`. Build it on a **Linux x86_64** host (a
native box or cloud VM is far better than emulated Docker for this) with the
Android NDK + the QNN SDK:

```bash
git clone --branch v1.2.0 https://github.com/pytorch/executorch.git
cd executorch && ./install_requirements.sh
export QNN_SDK_ROOT=/path/to/qairt/<version>     # the SDK ExecuTorch fetched, or QPM
export ANDROID_NDK=/path/to/android-ndk-r26d
# build the Android AAR with the QNN backend enabled:
EXECUTORCH_BUILD_QNN=ON ANDROID_ABIS=arm64-v8a \
  scripts/build_android_library.sh
# -> copy the resulting executorch.aar to <repo>/android/app/libs/executorch-qnn.aar
```
Best done with an on-site Qualcomm/Meta mentor, or on a cloud Ubuntu 22.04 x64 VM.

## Put it together + test on the S25 (your teammate)
1. Ensure all three artifacts are in place (1+2 from the Docker run, 3 from the AAR build).
2. `cd android && ./gradlew assembleDebug` → `app-debug.apk`.
3. On the S25: `adb install -r app-debug.apk`, open the app.
4. `adb logcat -s Slashh` → expect **`model loaded: stress_model_qnn.pte (NPU/QNN)`**.
   - If QNN fails to load, the app auto-falls back to CPU (demo never hard-fails).
5. Speak / tap **Simulate stress**; confirm it responds. For NPU proof, capture an
   on-device latency/utilization trace (or cite the AI Hub profile: 100% NPU, 0.06 ms).

## Demo-claim guidance
- After step 4 prints the QNN line **on the real S25**, you may say the **live app
  runs on the Snapdragon NPU**.
- Until then: "runs on-device via ExecuTorch (CPU today); same model profiled on
  the Hexagon NPU via AI Hub at 100% NPU / 0.06 ms."
