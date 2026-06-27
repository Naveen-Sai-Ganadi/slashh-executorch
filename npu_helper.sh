#!/system/bin/sh
# npu_helper.sh — shell-domain WavLM stage-2 NPU loop for the slashh background monitor.
#
# WHY THIS EXISTS: on a retail (locked) Galaxy S25, the app process
# (sec_untrusted_app) is SELinux-blocked from /dev/fastrpc-cdsp, so an in-app QNN
# delegate fails with error 4000. The shell domain (uid 2000) HAS cDSP access.
# This script runs in the shell domain and drives the Hexagon HTP on the app's
# behalf, exchanging audio + logits over a file channel in the app's external
# files dir (group ext_data_rw is shared by shell and the app — no INTERNET, no
# new app permission, preserving the app's on-device-only privacy stance).
#
# PROTOCOL (single-flight, app is producer of input / consumer of output):
#   app  ->  writes npu_in.raw (192000 bytes LE float32 = 48000 samples @16kHz),
#            deletes any stale npu_out.ready, then touches npu_in.ready
#   here ->  sees npu_in.ready, claims it (rm), copies npu_in.raw into the rig,
#            runs WavLM stage 2 on the NPU, writes npu_out.raw (4-byte logit)
#            atomically, then touches npu_out.ready
#   app  ->  polls npu_out.ready, reads the 4-byte logit, sigmoids it, then
#            deletes npu_out.ready + npu_out.raw
# On any failure the helper touches npu_err.ready so the app falls back to CPU
# fast instead of waiting for the timeout.
#
# USAGE (from host):
#   adb push npu_helper.sh /data/local/tmp/qnntest/npu_helper.sh
#   adb shell "chmod 755 /data/local/tmp/qnntest/npu_helper.sh"
#   adb shell "nohup /data/local/tmp/qnntest/npu_helper.sh > /data/local/tmp/qnntest/npu_helper.log 2>&1 &"
# Stop with:  adb shell "pkill -f npu_helper.sh"

RIG=/data/local/tmp/qnntest
CH=/sdcard/Android/data/ai.slashh/files
OUTRAW="$RIG/out2f/Result_0/output_aten_linear_default_170_0.raw"
POLL=0.15            # seconds between channel polls
EXPECT_BYTES=192000  # 48000 float32 samples

export LD_LIBRARY_PATH="$RIG"
export ADSP_LIBRARY_PATH="$RIG"
cd "$RIG" || { echo "FATAL: no rig dir $RIG"; exit 1; }

log() { echo "$(date '+%H:%M:%S') $*"; }

# sanity: the pieces the loop depends on
for f in qnn-net-run forward_2.bin libQnnHtp.so inlist2f.txt; do
  [ -e "$RIG/$f" ] || { log "FATAL: missing $RIG/$f"; exit 1; }
done

log "npu_helper up. rig=$RIG channel=$CH"
served=0
while :; do
  if [ -f "$CH/npu_in.ready" ]; then
    # claim the request (single-flight): remove the marker first so a slow run
    # can't be double-triggered by the same marker.
    rm -f "$CH/npu_in.ready"

    if [ ! -f "$CH/npu_in.raw" ]; then
      log "WARN ready set but npu_in.raw missing; skipping"
      : > "$CH/npu_err.ready" 2>/dev/null
      continue
    fi

    sz=$(stat -c %s "$CH/npu_in.raw" 2>/dev/null || echo 0)
    if [ "$sz" != "$EXPECT_BYTES" ]; then
      log "WARN npu_in.raw is $sz bytes (want $EXPECT_BYTES); skipping"
      : > "$CH/npu_err.ready" 2>/dev/null
      continue
    fi

    # stage the new audio window and run stage 2 on the NPU
    cp "$CH/npu_in.raw" "$RIG/wavlm_input.raw"
    rm -rf "$RIG/out2f"; mkdir -p "$RIG/out2f"
    if ./qnn-net-run --backend libQnnHtp.so --retrieve_context forward_2.bin \
         --input_list inlist2f.txt --output_dir out2f >/dev/null 2>&1 \
       && [ -f "$OUTRAW" ]; then
      # publish result atomically: write tmp, rename, then raise the flag
      cp "$OUTRAW" "$CH/.npu_out.tmp"
      mv "$CH/.npu_out.tmp" "$CH/npu_out.raw"
      : > "$CH/npu_out.ready"
      served=$((served+1))
      log "served #$served ($sz bytes -> logit)"
    else
      log "ERROR qnn-net-run failed"
      : > "$CH/npu_err.ready" 2>/dev/null
    fi
  fi
  sleep "$POLL"
done
