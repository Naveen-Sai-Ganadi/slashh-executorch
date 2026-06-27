#!/usr/bin/env bash
# Build the QNN-enabled ExecuTorch Android AAR (NPU artifact #3).
#
# BEST RUN ON A NATIVE LINUX x86_64 HOST (cloud Ubuntu 22.04 VM is ideal) — the
# C++ cross-compile is slow/fragile under macOS Docker emulation. Outputs
# android/app/app/libs/executorch-qnn.aar, which build.gradle.kts auto-uses.
#
# Prereqs on the host: git, cmake, ninja-build, python3.11, openjdk-17, unzip,
# curl, libc++1.  QAIRT SDK: this repo's .qairt/ (from build_qnn_pte.sh) or set
# QNN_SDK_ROOT yourself.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

# NDK r27+ makes 16 KB ELF alignment the default — required for the S25 Ultra
# (Android 15, 16 KB pages), else libexecutorch.so / libqnn_executorch_backend.so
# won't load on-device.
: "${ANDROID_NDK_VERSION:=r27c}"
: "${ET_VERSION:=v1.2.0}"
WORK="${WORK:-$HOME/et-qnn-build}"
mkdir -p "$WORK" && cd "$WORK"

# 1) QNN SDK — reuse repo .qairt/ if present, else download QAIRT Community (public, no login)
QNN_VER=2.37.0.250724
if [ -z "${QNN_SDK_ROOT:-}" ]; then
  QNN_SDK_ROOT="$(find "$REPO/.qairt" -maxdepth 3 -type d -name '2.37.0.*' 2>/dev/null | head -1 || true)"
fi
if [ -z "${QNN_SDK_ROOT:-}" ]; then
  echo "downloading QAIRT $QNN_VER ..."
  curl -L -o "$WORK/qairt.zip" \
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_VER}/v${QNN_VER}.zip"
  unzip -q "$WORK/qairt.zip" -d "$WORK/qairt-sdk"
  QNN_SDK_ROOT="$(find "$WORK/qairt-sdk" -maxdepth 3 -type d -name "${QNN_VER}" | head -1)"
fi
[ -n "${QNN_SDK_ROOT:-}" ] || { echo "QNN SDK not found"; exit 1; }
export QNN_SDK_ROOT
echo "QNN_SDK_ROOT=$QNN_SDK_ROOT"

# 2) Android NDK
if [ -z "${ANDROID_NDK:-}" ]; then
  if [ ! -d "android-ndk-${ANDROID_NDK_VERSION}" ]; then
    curl -L -o ndk.zip "https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION}-linux.zip"
    unzip -q ndk.zip && rm ndk.zip
  fi
  export ANDROID_NDK="$WORK/android-ndk-${ANDROID_NDK_VERSION}"
fi
echo "ANDROID_NDK=$ANDROID_NDK"

# 3) ExecuTorch source @ the version that produced the .pte
if [ ! -d executorch ]; then
  git clone --branch "$ET_VERSION" --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/executorch.git
fi
cd executorch
./install_requirements.sh || pip install -r requirements-dev.txt || true

# 4) Build the AAR with the QNN backend.
# The S25 Ultra uses 16 KB pages; NDK r27 does NOT align to 16 KB by default and
# executorch's build script ignores CMAKE_ARGS, so inject the linker flag right
# into its cmake invocation (idempotent), and clean cmake-out so it takes effect.
if ! grep -q "max-page-size=16384" scripts/build_android_library.sh; then
  sed -i 's#cmake \. -DCMAKE_INSTALL_PREFIX#cmake . -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384" -DCMAKE_INSTALL_PREFIX#g' \
    scripts/build_android_library.sh
fi
rm -rf cmake-out-android-* cmake-out-android-so 2>/dev/null || true
export EXECUTORCH_BUILD_QNN=ON
export ANDROID_ABIS=arm64-v8a
bash scripts/build_android_library.sh

# 5) Deliver
AAR="$(find . -name 'executorch*.aar' | head -1)"
[ -n "$AAR" ] || { echo "AAR not produced — check build log"; exit 1; }
mkdir -p "$REPO/android/app/libs"
cp "$AAR" "$REPO/android/app/libs/executorch-qnn.aar"
echo "OK -> android/app/libs/executorch-qnn.aar  (build.gradle.kts will auto-use it)"
