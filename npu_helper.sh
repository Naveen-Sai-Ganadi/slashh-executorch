#!/system/bin/sh
# npu_helper.sh — shell-domain WavLM stage-2 NPU loop for the slashh background monitor.
# Patched: chmod 0666 the published channel files so the app (different uid, FUSE mount)
# can read them.

RIG=/data/local/tmp/qnntest
CH=/sdcard/Android/data/ai.slashh/files
OUTRAW="$RIG/out2f/Result_0/output_aten_linear_default_170_0.raw"
POLL=0.15            # seconds between channel polls
EXPECT_BYTES=192000  # 48000 float32 samples

export LD_LIBRARY_PATH="$RIG"
export ADSP_LIBRARY_PATH="$RIG"
cd "$RIG" || { echo "FATAL: no rig dir $RIG"; exit 1; }

log() { echo "$(date '+%H:%M:%S') $*"; }

for f in qnn-net-run forward_2.bin libQnnHtp.so inlist2f.txt; do
  [ -e "$RIG/$f" ] || { log "FATAL: missing $RIG/$f"; exit 1; }
done

log "npu_helper up (chmod-patched). rig=$RIG channel=$CH"
served=0
while :; do
  if [ -f "$CH/npu_in.ready" ]; then
    rm -f "$CH/npu_in.ready"

    if [ ! -f "$CH/npu_in.raw" ]; then
      log "WARN ready set but npu_in.raw missing; skipping"
      : > "$CH/npu_err.ready" 2>/dev/null; chmod 0666 "$CH/npu_err.ready" 2>/dev/null
      continue
    fi

    sz=$(stat -c %s "$CH/npu_in.raw" 2>/dev/null || echo 0)
    if [ "$sz" != "$EXPECT_BYTES" ]; then
      log "WARN npu_in.raw is $sz bytes (want $EXPECT_BYTES); skipping"
      : > "$CH/npu_err.ready" 2>/dev/null; chmod 0666 "$CH/npu_err.ready" 2>/dev/null
      continue
    fi

    cp "$CH/npu_in.raw" "$RIG/wavlm_input.raw"
    rm -rf "$RIG/out2f"; mkdir -p "$RIG/out2f"
    if ./qnn-net-run --backend libQnnHtp.so --retrieve_context forward_2.bin \
         --input_list inlist2f.txt --output_dir out2f >/dev/null 2>&1 \
       && [ -f "$OUTRAW" ]; then
      cp "$OUTRAW" "$CH/.npu_out.tmp"
      mv "$CH/.npu_out.tmp" "$CH/npu_out.raw"
      chmod 0666 "$CH/npu_out.raw" 2>/dev/null
      : > "$CH/npu_out.ready"
      chmod 0666 "$CH/npu_out.ready" 2>/dev/null
      served=$((served+1))
      log "served #$served ($sz bytes -> logit)"
    else
      log "ERROR qnn-net-run failed"
      : > "$CH/npu_err.ready" 2>/dev/null; chmod 0666 "$CH/npu_err.ready" 2>/dev/null
    fi
  fi
  sleep "$POLL"
done
