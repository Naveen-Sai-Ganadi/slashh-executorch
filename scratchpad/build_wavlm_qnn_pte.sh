#!/usr/bin/env bash
# Produce the WavLM QNN/Hexagon .pte for the S25 from macOS, via a Linux x86_64
# container (ExecuTorch's QNN backend is Linux-x86-only and auto-downloads the
# QNN SDK there). Adds transformers+peft (to rebuild/merge WavLM-large) over the
# StressNet recipe, and mounts the HF cache so wavlm-large is NOT re-downloaded
# through the slow emulated network. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
HF_ARGS=()
if [ -d "$HF_CACHE" ]; then
  echo "mounting HF cache: $HF_CACHE (skips wavlm-large download)"
  HF_ARGS=(-v "$HF_CACHE":/root/.cache/huggingface)
else
  echo "WARNING: no HF cache at $HF_CACHE; wavlm-large will download in-container (slow)"
fi

# Reuse a pre-downloaded QAIRT SDK if present (faster than emulated-net download).
SDK_ARGS=()
SDK_DIR="$(find .qairt -maxdepth 3 -type d -name '2.37.0.*' 2>/dev/null | head -1 || true)"
if [ -n "${SDK_DIR}" ]; then
  echo "using pre-downloaded QAIRT SDK: ${SDK_DIR}"
  SDK_ARGS=(-v "$PWD/${SDK_DIR}":/qairt -e QNN_SDK_ROOT=/qairt)
fi

docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work \
  -v slashh-pipcache:/root/.cache/pip \
  -v slashh-etcache:/root/.cache/executorch \
  "${HF_ARGS[@]+"${HF_ARGS[@]}"}" "${SDK_ARGS[@]+"${SDK_ARGS[@]}"}" \
  -e N_CALIB="${N_CALIB:-8}" \
  -e FP16_ONLY="${FP16_ONLY:-}" \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e MALLOC_ARENA_MAX="2" \
  -e MALLOC_TRIM_THRESHOLD_="0" \
  python:3.11-slim bash -c '
    set -e
    echo "[deps] system";
    apt-get update -qq && apt-get install -y -qq libsndfile1 libgomp1 libc++1 libc++abi1 >/dev/null;
    echo "[deps] pip executorch 1.2.0 + torch 2.11.0 + transformers + peft (emulated x86 — slow, cached after first run)";
    pip install --no-input executorch==1.2.0 torch==2.11.0 torchaudio==2.11.0 \
        transformers peft soundfile numpy 2>&1 | tail -2;
    echo "[run] export WavLM QNN .pte";
    export LD_LIBRARY_PATH="${QNN_SDK_ROOT:-/qairt}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}";
    python scratchpad/export_wavlm_qnn_pte.py
  '
