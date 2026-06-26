#!/bin/sh
# Run the Android audio-core unit tests on a plain JVM — no Android SDK, no
# Gradle, no device. The audio pipeline (AudioConfig, RealDft, LogMel, Vad,
# StressPipeline + any android-free UI view-models) is pure Kotlin/JVM, so the
# parity gate (LogMelParityTest) and logic tests run anywhere kotlinc + a JDK
# exist. The device-only classes (AudioCapture, ExecuTorch wrapper, Activities)
# are skipped automatically — they import android.* / org.pytorch.
#
# Usage:  sh android/run_jvm_tests.sh
# Requires: kotlinc and a JDK (brew: `brew install kotlin`). Test jars are
# fetched once into android/.jvmtest/ (gitignored).
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
APP="$ROOT/android/app/src"
WORK="$ROOT/android/.jvmtest"
LIB="$WORK/libs"
OUT="$WORK/out"
mkdir -p "$LIB" "$OUT"
rm -rf "$OUT"/*

# --- locate toolchain -------------------------------------------------------
KOTLINC=$(command -v kotlinc || echo /opt/homebrew/bin/kotlinc)
[ -x "$KOTLINC" ] || { echo "kotlinc not found (try: brew install kotlin)"; exit 127; }
# the macOS /usr/bin/java is a stub; prefer a real JDK
for j in /opt/homebrew/opt/openjdk/bin/java "$JAVA_HOME/bin/java" "$(command -v java 2>/dev/null)"; do
  if [ -x "$j" ] && "$j" -version >/dev/null 2>&1; then JAVA="$j"; break; fi
done
[ -n "$JAVA" ] || { echo "no working JDK (try: brew install openjdk)"; exit 127; }
KLIB=$(dirname "$(dirname "$(readlink -f "$KOTLINC" 2>/dev/null || echo "$KOTLINC")")")/libexec/lib
[ -f "$KLIB/kotlin-stdlib.jar" ] || KLIB=/opt/homebrew/opt/kotlin/libexec/lib

# --- fetch test jars once ---------------------------------------------------
fetch() { [ -f "$LIB/$2" ] || curl -fsSL -o "$LIB/$2" "$1"; }
fetch https://repo1.maven.org/maven2/junit/junit/4.13.2/junit-4.13.2.jar junit-4.13.2.jar
fetch https://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar hamcrest-core-1.3.jar
fetch https://repo1.maven.org/maven2/org/json/json/20240303/json-20240303.jar json-20240303.jar
CP="$LIB/junit-4.13.2.jar:$LIB/hamcrest-core-1.3.jar:$LIB/json-20240303.jar"

# --- collect sources --------------------------------------------------------
# main: only android-free Kotlin (skip files importing android.* or org.pytorch)
MAIN=""
for f in $(find "$APP/main/java" -name '*.kt'); do
  grep -Eq '^import (android|org\.pytorch)' "$f" || MAIN="$MAIN $f"
done
# test: all of them are pure JVM (JUnit + org.json)
TEST=$(find "$APP/test/java" -name '*.kt')

echo "[jvm-tests] compiling $(echo $MAIN $TEST | wc -w | tr -d ' ') Kotlin files…"
# shellcheck disable=SC2086
"$KOTLINC" $MAIN $TEST -cp "$CP" -d "$OUT" 2>&1 | grep -v 'warning: redundant' || true

# --- discover test classes (top-level classes in test/) ---------------------
CLASSES=$(find "$APP/test/java" -name '*.kt' -exec grep -h '^class ' {} \; \
          | sed -E 's/^class ([A-Za-z0-9_]+).*/\1/')
PKG=ai.slashh.audio
FQ=$(for c in $CLASSES; do echo "$PKG.$c"; done)

echo "[jvm-tests] running: $FQ"
# shellcheck disable=SC2086
"$JAVA" -cp "$OUT:$APP/test/resources:$CP:$KLIB/kotlin-stdlib.jar" \
  org.junit.runner.JUnitCore $FQ
