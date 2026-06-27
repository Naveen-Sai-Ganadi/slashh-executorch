#!/usr/bin/env bash
# Build the QNN-enabled ExecuTorch AAR from macOS via a linux/amd64 container
# (when no native Linux box / WSL is available). SLOW under x86 emulation; the
# native path (tools/qnn/build_qnn_aar.sh on Ubuntu/WSL) is faster. Reuses the
# pre-downloaded QAIRT in .qairt/ and caches the build tree + pip across runs.
set -euo pipefail
cd "$(dirname "$0")/../.."

docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work \
  -v slashh-aarwork:/root/et-qnn-build \
  -v slashh-pipcache:/root/.cache/pip \
  ubuntu:22.04 bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    echo "[deps] apt";
    apt-get update -qq && apt-get install -y -qq \
      git cmake ninja-build python3 python3-pip python3-venv \
      openjdk-17-jdk-headless unzip curl zip ca-certificates libc++1 >/dev/null;
    echo "[build] tools/qnn/build_qnn_aar.sh";
    bash tools/qnn/build_qnn_aar.sh
  '
