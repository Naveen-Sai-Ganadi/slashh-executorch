#!/usr/bin/env bash
# Produce the QNN/Hexagon .pte for the S25 from macOS, via a Linux x86_64
# container (ExecuTorch's QNN backend is Linux-x86-only and auto-downloads the
# QNN SDK there — no Qualcomm login needed). Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

# If QAIRT was pre-downloaded on the host (faster/reliable than the container's
# emulated network), mount it and point QNN_SDK_ROOT at it so ExecuTorch skips
# its own download. Expect it extracted under .qairt/ (qairt/<version>/).
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
  "${SDK_ARGS[@]}" \
  python:3.11-slim bash -c '
    set -e
    echo "[1/3] system deps";
    apt-get update -qq && apt-get install -y -qq libsndfile1 libgomp1 libc++1 libc++abi1 >/dev/null;
    echo "[2/3] pip install executorch 1.2.0 + torch 2.11.0 (large, emulated x86 — slow, cached after first run)";
    pip install --no-input executorch==1.2.0 torch==2.11.0 torchaudio==2.11.0 soundfile numpy 2>&1 | tail -1;
    echo "[3/3] export QNN .pte + extract QNN runtime libs";
    export LD_LIBRARY_PATH="${QNN_SDK_ROOT:-/qairt}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}";
    python tools/qnn/export_qnn_pte.py
  '
