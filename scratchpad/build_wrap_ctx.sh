#!/usr/bin/env bash
# Path B: wrap the AI Hub QNN context binary into an ExecuTorch .pte, from macOS
# via a Linux x86_64 container (ExecuTorch's QNN backend is Linux-x86-only).
# Uses the prebaked wavlm-qnn:latest image (torch+executorch --no-deps already
# installed) so each run starts instantly instead of reinstalling under emulation.
# The QAIRT SDK is mounted from the host (.qairt/...) and QNN_SDK_ROOT is set so
# the qualcomm backend's import-time auto-download is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

SDK_DIR="$(find .qairt -maxdepth 3 -type d -name '2.37.0.*' 2>/dev/null | head -1 || true)"
if [ -z "${SDK_DIR}" ]; then
  echo "FATAL: QAIRT SDK not found under .qairt/ (expected .qairt/qairt/2.37.0.*)" >&2
  exit 1
fi
echo "using QAIRT SDK: ${SDK_DIR}"

docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work \
  -v "$PWD/${SDK_DIR}":/qairt:ro \
  -e QNN_SDK_ROOT=/qairt \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e MALLOC_ARENA_MAX=2 \
  wavlm-qnn:latest bash -c '
    set -eo pipefail
    export LD_LIBRARY_PATH="/qairt/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
    echo "[env] QNN_SDK_ROOT=$QNN_SDK_ROOT"
    echo "[env] libQnnHtp.so: $(ls -la /qairt/lib/x86_64-linux-clang/libQnnHtp.so 2>&1 | head -1)"
    echo "[run] wrap AI Hub ctx binary -> ExecuTorch .pte"
    python scratchpad/wrap_wavlm_ctx_pte.py
  '
