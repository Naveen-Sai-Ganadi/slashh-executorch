#!/usr/bin/env bash
# ============================================================================
# npu_helper_whisper.sh — start/stop the on-device Whisper NPU daemon for Slashh.
#
# The retail Galaxy S25's SELinux policy blocks the app (sec_untrusted_app) from
# the Hexagon cDSP, so Whisper runs OUT OF PROCESS in the shell domain: this
# script launches `whisper_qnn --watch`, which loads the encoder/decoder QNN
# context binaries ONCE and serves the Slashh app over a file channel in the
# app's external files dir. This is the SAME out-of-process pattern the WavLM
# stress rig (npu_helper.sh) already uses; only the channel markers differ
# (whisper_* vs npu_*), so the two daemons coexist on one device.
#
# Channel = /sdcard/Android/data/ai.slashh/files (the app's getExternalFilesDir),
# matching ai.slashh.runtime.WhisperConfig. No INTERNET, audio never leaves the
# device. Ported from the ScamShield/Edge whisper rig; only CH (and the package)
# changed.
#
# Usage: ./npu_helper_whisper.sh [start|stop|status|log]
# Env overrides: ADB, ANDROID_SERIAL, QNN (QAIRT 2.45 SDK dir — only needed the
#   first time, to stage the QNN runtime libs; they then stay resident in $RIG).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
S="${ANDROID_SERIAL:-$("$ADB" devices | awk 'NR==2{print $1}')}"
RIG=/data/local/tmp/whisper_rig
CH=/sdcard/Android/data/ai.slashh/files
QNN="${QNN:-}"
ART="$HERE/whisper_tiny_qnn"
DAEMON="whisper_qnn --watch"

sh_(){ "$ADB" -s "$S" shell "$@"; }
push_if(){ [ -f "$1" ] && "$ADB" -s "$S" push "$1" "$RIG/" >/dev/null && echo "  pushed $(basename "$1")" || true; }
# Push a rig file only if it isn't already on the device (the big model bins are
# ~117 MB total — don't re-push every start).
push_missing(){
  local name; name="$(basename "$1")"
  if sh_ "[ -s $RIG/$name ]" 2>/dev/null; then echo "  have $name"; else push_if "$1"; fi
}

case "${1:-start}" in
  start)
    [ -z "$S" ] && { echo "no device"; exit 1; }
    echo "device: $S"
    sh_ "mkdir -p $RIG $CH"
    echo "staging rig (idempotent)..."
    # QNN runtime libs: only if a local QAIRT 2.45 SDK is provided AND they aren't
    # already resident. On a device that already ran the rig, this is a no-op.
    if [ -n "$QNN" ]; then
      for f in libQnnHtp.so libQnnHtpPrepare.so libQnnHtpV79Stub.so libQnnSystem.so; do
        sh_ "[ -s $RIG/$f ]" 2>/dev/null && echo "  have $f" || push_if "$QNN/lib/aarch64-android/$f"
      done
      sh_ "[ -s $RIG/libQnnHtpV79Skel.so ]" 2>/dev/null && echo "  have libQnnHtpV79Skel.so" \
        || push_if "$QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"
    fi
    push_missing "$ART/encoder.bin"
    push_missing "$ART/decoder.bin"
    push_if "$HERE/vocab.bin"
    push_if "$HERE/mel_filters.bin"
    push_if "$HERE/whisper_qnn"
    echo "launching daemon (persistent, survives adb disconnect)..."
    sh_ "pkill -f '$DAEMON' 2>/dev/null; sleep 0.4; cd $RIG && chmod +x whisper_qnn && \
         LD_LIBRARY_PATH=$RIG ADSP_LIBRARY_PATH=$RIG nohup ./whisper_qnn --watch \
           encoder.bin decoder.bin vocab.bin mel_filters.bin $CH libQnnHtp.so \
           > $RIG/whisper_watch.log 2>&1 < /dev/null & \
         sleep 2; echo '--- log ---'; tail -6 $RIG/whisper_watch.log; \
         pgrep -f '$DAEMON' >/dev/null && echo 'STATUS: RUNNING' || echo 'STATUS: FAILED'"
    ;;
  stop)   sh_ "pkill -f '$DAEMON' && echo stopped || echo 'not running'";;
  status) sh_ "pgrep -f '$DAEMON' >/dev/null && echo RUNNING || echo STOPPED; echo '--- log ---'; tail -10 $RIG/whisper_watch.log 2>/dev/null";;
  log)    sh_ "tail -40 $RIG/whisper_watch.log 2>/dev/null";;
  *) echo "usage: $0 [start|stop|status|log]"; exit 2;;
esac
