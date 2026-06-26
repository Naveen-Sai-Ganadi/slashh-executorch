#!/usr/bin/env bash
# Produce the QNN/Hexagon .pte for the S25 from macOS, via a Linux x86_64
# container (ExecuTorch's QNN backend is Linux-x86-only and auto-downloads the
# QNN SDK there — no Qualcomm login needed). Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work \
  -v slashh-pipcache:/root/.cache/pip \
  -v slashh-etcache:/root/.cache/executorch \
  python:3.11-slim bash -c '
    set -e
    echo "[1/3] system deps";
    apt-get update -qq && apt-get install -y -qq libsndfile1 libgomp1 >/dev/null;
    echo "[2/3] pip install executorch 1.2.0 + torch 2.11.0 (large, emulated x86 — slow, cached after first run)";
    pip install --no-input executorch==1.2.0 torch==2.11.0 torchaudio==2.11.0 soundfile numpy 2>&1 | tail -1;
    echo "[3/3] export QNN .pte + extract QNN runtime libs (auto-downloads QNN SDK on first import)";
    python tools/qnn/export_qnn_pte.py
  '
