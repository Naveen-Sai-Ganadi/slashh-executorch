#!/usr/bin/env bash
# ============================================================================
# build.sh — cross-compile whisper_qnn for Android arm64-v8a (Hexagon HTP NPU),
#            then print the adb push + on-device run commands.
#
# Direct clang++ invocation (no CMake required). The QNN backend libraries are
# loaded at runtime via dlopen(); only the QNN headers are needed to compile and
# -ldl to link.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Toolchain / SDK paths (override via env if your layout differs) --------
QNN="${QNN:-/private/tmp/claude-501/-Users-sumakatabattuni-Documents-Personal-slashh-edge/0573850d-791b-4a19-ab79-95d1c47bf752/scratchpad/qairt245/qairt/2.45.0.260326}"
NDK="${NDK:-/Users/sumakatabattuni/Library/Android/sdk/ndk/26.3.11579264}"
TC="$NDK/toolchains/llvm/prebuilt/darwin-x86_64"
CLANGXX="$TC/bin/aarch64-linux-android30-clang++"

ADB="${ADB:-/Users/sumakatabattuni/Library/Android/sdk/platform-tools/adb}"
RIG="${RIG:-/data/local/tmp/whisper_rig}"

OUT="$HERE/whisper_qnn"

# ---- Sanity checks ---------------------------------------------------------
if [[ ! -x "$CLANGXX" ]]; then
  echo "[ERROR] NDK clang++ not found at: $CLANGXX" >&2
  echo "        Install it with:" >&2
  echo "        \$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager \"ndk;26.3.11579264\"" >&2
  exit 1
fi
if [[ ! -d "$QNN/include/QNN" ]]; then
  echo "[ERROR] QNN headers not found at: $QNN/include/QNN" >&2
  exit 1
fi

# ---- Compile ---------------------------------------------------------------
echo "[build] Compiling whisper_qnn (arm64-v8a, android-30)..."
"$CLANGXX" \
  -std=c++17 -O3 -fPIC -fvisibility=hidden \
  -march=armv8.2-a+fp16 \
  -DQNN_API='__attribute__((visibility("default")))' \
  -I"$QNN/include/QNN" \
  -static-libstdc++ \
  "$HERE/whisper_qnn.cpp" -o "$OUT" \
  -ldl -lm -llog

echo "[build] OK -> $OUT"
file "$OUT" || true

# ---- Print the on-device deploy + run commands -----------------------------
cat <<EOF

================================================================================
 On-device deploy + run (Galaxy S25 Ultra, SM8750, Hexagon V79)
================================================================================
# 1) Create the rig dir on the phone
$ADB shell "mkdir -p $RIG"

# 2) Push the runner
$ADB push "$OUT" $RIG

# 3) Push the QNN runtime libs (CPU-side: HTP core + V79 stub + prepare + system)
$ADB push "$QNN/lib/aarch64-android/libQnnHtp.so"          $RIG
$ADB push "$QNN/lib/aarch64-android/libQnnHtpPrepare.so"   $RIG
$ADB push "$QNN/lib/aarch64-android/libQnnHtpV79Stub.so"   $RIG
$ADB push "$QNN/lib/aarch64-android/libQnnSystem.so"       $RIG

# 4) Push the Hexagon DSP skel for V79 (runs on the NPU; found via ADSP_LIBRARY_PATH)
$ADB push "$QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" $RIG

# 5) Push the model context binaries + the front-end assets
$ADB push "$HERE/../whisper_tiny_qnn/whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_8_elite_for_galaxy/encoder.bin" $RIG
$ADB push "$HERE/../whisper_tiny_qnn/whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_8_elite_for_galaxy/decoder.bin" $RIG
$ADB push "$HERE/mel_filters.bin" $RIG     # 201*80 float32 mel filterbank (64320 B)
$ADB push "$HERE/vocab.bin"       $RIG     # 50257 byte-level BPE tokens for detok

# 6a) Legacy mode — precomputed fp16 features -> token IDs
$ADB push /path/to/input_features.raw $RIG
$ADB shell "cd $RIG && \\
  export LD_LIBRARY_PATH=$RIG && \\
  export ADSP_LIBRARY_PATH=$RIG && \\
  chmod +x ./whisper_qnn && \\
  ./whisper_qnn encoder.bin decoder.bin input_features.raw libQnnHtp.so"

# 6b) PCM mode — raw mono float32 LE PCM -> TEXT (mel + detok on-device)
$ADB push /path/to/window.raw $RIG
$ADB shell "cd $RIG && export LD_LIBRARY_PATH=$RIG && export ADSP_LIBRARY_PATH=$RIG && \\
  ./whisper_qnn --pcm encoder.bin decoder.bin vocab.bin mel_filters.bin window.raw libQnnHtp.so"

# 6c) Daemon mode — resident; serves the app's file channel forever (~150 ms/req)
CHAN=/sdcard/Android/data/com.scamshield.app/files
$ADB shell "cd $RIG && export LD_LIBRARY_PATH=$RIG && export ADSP_LIBRARY_PATH=$RIG && \\
  ./whisper_qnn --watch encoder.bin decoder.bin vocab.bin mel_filters.bin \$CHAN libQnnHtp.so"

# Validation only (no NPU): mel front-end vs Python WhisperFeatureExtractor
#   ./whisper_qnn --dump-mel mel_filters.bin window.raw out.raw   # 480000 B fp16
================================================================================
EOF
