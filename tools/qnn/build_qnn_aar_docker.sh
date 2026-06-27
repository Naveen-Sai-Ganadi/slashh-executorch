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
  -v slashh-androidsdk:/opt/android-sdk \
  ubuntu:22.04 bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    echo "[deps] apt";
    apt-get update -qq && apt-get install -y -qq \
      git cmake ninja-build python3 python3-pip python3-venv \
      openjdk-17-jdk-headless unzip curl zip ca-certificates libc++1 >/dev/null;

    echo "[sdk] Android SDK (cmdline-tools + platform + build-tools)";
    export ANDROID_HOME=/opt/android-sdk ANDROID_SDK_ROOT=/opt/android-sdk;
    if [ ! -x "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" ]; then
      mkdir -p "$ANDROID_HOME/cmdline-tools";
      curl -sL -o /tmp/clt.zip https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip;
      unzip -q /tmp/clt.zip -d "$ANDROID_HOME/cmdline-tools";
      mv "$ANDROID_HOME/cmdline-tools/cmdline-tools" "$ANDROID_HOME/cmdline-tools/latest";
    fi;
    export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH";
    yes | sdkmanager --licenses >/dev/null 2>&1 || true;
    sdkmanager --install "platform-tools" "platforms;android-34" "platforms;android-35" "build-tools;34.0.0" >/dev/null;

    echo "[build] tools/qnn/build_qnn_aar.sh";
    bash tools/qnn/build_qnn_aar.sh
  '
