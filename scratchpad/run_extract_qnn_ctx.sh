#!/usr/bin/env bash
# Extract the raw QNN context binary from the WavLM .pte inside a linux/amd64
# container (ExecuTorch's native PyQnnManagerAdaptor is x86-linux only).
# Reuses the pip cache volume so executorch/torch wheels are not re-downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."

SDK_DIR="$(find .qairt -maxdepth 4 -type d -name '2.37.0.*' 2>/dev/null | head -1 || true)"
SDK_ARGS=()
if [ -n "${SDK_DIR}" ]; then
  echo "using QAIRT SDK: ${SDK_DIR}"
  SDK_ARGS=(-v "$PWD/${SDK_DIR}":/qairt -e QNN_SDK_ROOT=/qairt)
else
  echo "WARNING: no QAIRT SDK found under .qairt; StripProtocol may still work (pure byte op)"
fi

docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work \
  -v slashh-pipcache:/root/.cache/pip \
  "${SDK_ARGS[@]+"${SDK_ARGS[@]}"}" \
  python:3.11-slim bash -c '
    set -e
    echo "[deps] system";
    apt-get update -qq && apt-get install -y -qq libgomp1 libc++1 libc++abi1 >/dev/null;
    echo "[deps] pip executorch 1.2.0 + torch 2.11.0 (cached after first run)";
    pip install --no-input executorch==1.2.0 torch==2.11.0 numpy 2>&1 | tail -2;
    export LD_LIBRARY_PATH="${QNN_SDK_ROOT:-/qairt}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}";
    echo "[run] extract QNN context binary";
    python scratchpad/extract_qnn_ctx.py assets/teacher_wavlm_broad_qnn.pte
  '
