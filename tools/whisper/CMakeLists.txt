# ============================================================================
# CMakeLists.txt — cross-compile whisper_qnn for Android arm64-v8a (HTP NPU).
#
# The QNN backend libraries are NOT linked at build time; they are dlopen()'d
# at runtime (libQnnHtp.so / libQnnSystem.so). We only need the QNN headers to
# compile and -ldl to link.
#
# Build:
#   cmake -S . -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-30 \
#     -DANDROID_STL=c++_static \
#     -DQNN_SDK=/path/to/qairt/2.45.0.260326/include/QNN
#   cmake --build build-android -j
# ============================================================================
cmake_minimum_required(VERSION 3.18)
project(whisper_qnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK include dir (the directory containing QnnInterface.h etc.). Overridable
# on the command line with -DQNN_SDK=...
set(QNN_SDK
    "/private/tmp/claude-501/-Users-sumakatabattuni-Documents-Personal-slashh-edge/0573850d-791b-4a19-ab79-95d1c47bf752/scratchpad/qairt245/qairt/2.45.0.260326/include/QNN"
    CACHE PATH "Path to the QNN SDK include/QNN directory")

add_executable(whisper_qnn whisper_qnn.cpp)

target_include_directories(whisper_qnn PRIVATE ${QNN_SDK})

target_compile_options(whisper_qnn PRIVATE
    -O3
    -fPIC
    -fvisibility=hidden
    # _Float16 requires ARMv8.2-A FP16; the Snapdragon 8 Elite (Cortex-X/A7xx)
    # supports it. The android-30 arm64 clang enables fp16 by default, but make
    # it explicit so the build is reproducible across NDK versions.
    -march=armv8.2-a+fp16)

# QNN_API must be a default-visibility attribute so the dlopen'd providers are
# resolvable; matches the SDK's own build flags.
target_compile_definitions(whisper_qnn PRIVATE
    "QNN_API=__attribute__((visibility(\"default\")))")

# Only -ldl (and -lm/-llog) are needed; QNN libs are loaded at runtime.
target_link_libraries(whisper_qnn PRIVATE ${CMAKE_DL_LIBS} m log)
