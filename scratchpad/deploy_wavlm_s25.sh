#!/bin/sh
# Deploy the WavLM QNN .pte to the connected Samsung Galaxy S25 Ultra and run it
# on the Snapdragon NPU through the s25-dev app, then capture logcat proof that
# the graph executed on the Hexagon HTP (not a CPU fallback).
#
# Prereqs (all verified 2026-06-27):
#   - device: SM-S938U (S25 Ultra), SoC SM8750 / HTP V79, Android 16, arm64-v8a
#   - assets/teacher_wavlm_broad_qnn.pte built for QcomChipset.SM8750 (QNN 2.37)
#   - s25-dev app: executorch-qnn.aar + jniLibs libQnn*V79*.so all AISW 2.37.0
#   - MainActivity prefers the WavLM .pte in getExternalFilesDir(null)
#
# Usage:  sh scratchpad/deploy_wavlm_s25.sh
set -e

ADB=/Users/chinnu/Library/Android/sdk/platform-tools/adb
PTE=/Users/chinnu/slashh-executorch/slashh-executorch/assets/teacher_wavlm_broad_qnn.pte
ANDROID=/Users/chinnu/slashh-executorch/s25-dev/android
PKG=ai.slashh
DEST=/sdcard/Android/data/$PKG/files/teacher_wavlm_broad_qnn.pte

[ -f "$PTE" ] || { echo "FATAL: $PTE not built yet"; exit 1; }
echo "[1/5] .pte size: $(ls -la "$PTE" | awk '{print $5}') bytes"

echo "[2/5] push .pte -> device ($DEST)"
"$ADB" push "$PTE" "$DEST"
"$ADB" shell "ls -la $DEST"

echo "[3/5] build debug APK (gradle, host JVM)"
cd "$ANDROID"
./gradlew --no-daemon :app:assembleDebug
APK=$(find "$ANDROID/app/build/outputs/apk/debug" -name "*.apk" | head -1)
echo "      APK: $APK"

echo "[4/5] install + launch"
"$ADB" install -r "$APK"
"$ADB" logcat -c
"$ADB" shell am start -n "$PKG/.MainActivity"

echo "[5/5] capture logcat (15s) — looking for scorer + QNN/Hexagon delegation"
"$ADB" logcat -v time > /private/tmp/claude-501/-Users-chinnu-slashh-executorch-slashh-executorch/1ec10f14-c363-4f9a-9b54-de61dfb22ba1/scratchpad/s25_logcat.txt &
LCPID=$!
sleep 15
kill $LCPID 2>/dev/null || true
echo "--- scorer selection ---"
grep -E "Slashh.*scorer|Slashh.*model loaded|Slashh.*WavLM" /private/tmp/claude-501/-Users-chinnu-slashh-executorch-slashh-executorch/1ec10f14-c363-4f9a-9b54-de61dfb22ba1/scratchpad/s25_logcat.txt || true
echo "--- QNN/HTP delegation evidence ---"
grep -iE "Qnn|Htp|Hexagon|delegate|fastrpc|QnnExecuTorch" /private/tmp/claude-501/-Users-chinnu-slashh-executorch-slashh-executorch/1ec10f14-c363-4f9a-9b54-de61dfb22ba1/scratchpad/s25_logcat.txt | head -40 || true
echo "done."
