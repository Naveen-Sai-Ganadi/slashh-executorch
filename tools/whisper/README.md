# whisper_qnn — standalone QNN HTP runner for Whisper-tiny on the Hexagon NPU

A single C++ binary that runs Qualcomm AI Hub's **Whisper-tiny** (separate
encoder + decoder QNN context binaries) on the Snapdragon **Hexagon V79 NPU**
(HTP backend) of a Galaxy S25 Ultra, via the **QNN C API**.

It loads both context binaries **once**, runs the encoder, then the
autoregressive greedy decode loop entirely on the NPU. It now also includes a
hand-written **mel front-end** (PCM → log-Mel `input_features`) and a
**detokenizer** (token IDs → UTF-8 text via `vocab.bin`), so the full
PCM-in → text-out pipeline runs on-device with audio never leaving the phone. A
resident **`--watch` daemon** keeps both contexts loaded and serves the Android
app over a simple file channel.

### Modes

```
# Legacy positional mode — prints decoded token IDs (one space-separated line).
whisper_qnn <encoder.bin> <decoder.bin> <input_features.raw> [backend=libQnnHtp.so]

# PCM mode — PCM (mono float32 LE, any length) -> mel -> encoder -> decode ->
# detokenize -> print TEXT to stdout.
whisper_qnn --pcm <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <pcm.raw> [backend]

# Validation hook — compute the fp16 mel [1,80,3000] (480000 bytes) and exit.
# No NPU; diff against the Python WhisperFeatureExtractor reference.
whisper_qnn --dump-mel <mel_filters.bin> <pcm.raw> <out.raw>

# Daemon mode — load contexts + assets ONCE, then poll <channel_dir> every 50 ms
# and transcribe each request the app stages. Loops forever (~150 ms/request).
whisper_qnn --watch <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <channel_dir> [backend]
```

The mel front-end and detokenizer are pure C++ — no extra libraries. The two
assets they need ship alongside the binary:

| Asset | Size | Layout |
| --- | --- | --- |
| `mel_filters.bin` | 64320 B | `201 × 80` float32, row-major `[freq_bin][mel]` |
| `vocab.bin` | ~357 KB | 50257 token byte-strings separated by `\0` (byte-level BPE) |

### Mel front-end (matches OpenAI Whisper / `WhisperFeatureExtractor`)

`sample_rate=16000, n_fft=400, hop=160, n_mels=80, n_samples=480000` (30 s):
zero-pad/truncate PCM to 480000 → reflect-pad 200 each side (torch
`center=True`) → periodic Hann (`0.5 - 0.5·cos(2πn/400)`) → 3001 frames, drop the
last → 3000 → direct one-sided DFT (201 bins, precomputed cos/sin tables) →
`power = re² + im²` → `mel = melFilters·power` → `log10(max(mel, 1e-10))` → clamp
to `globalMax − 8.0` → `(x + 4.0)/4.0`. Emitted as fp16 row-major `[mel][time] =
[1,80,3000]`, straight into the encoder's `input_features` buffer. Validated
against the torch/`WhisperFeatureExtractor` reference to fp16-rounding precision
(max abs diff ≈ 1 ULP).

### Detokenizer

Concatenate the raw bytes of each non-special token (`id < 50257`); special
tokens (`id ≥ 50257`: EOT 50257 / SOT 50258 / `<|en|>` 50259 / `<|transcribe|>`
50359 / `<|notimestamps|>` 50363 / timestamps) carry no bytes and are skipped.
Trim a single leading space. Byte-level BPE makes the concatenation valid UTF-8.
Validated to match `WhisperTokenizer.decode(skip_special_tokens=True)` exactly.

### `--watch` channel contract

The app and daemon rendezvous through five files in `<channel_dir>` (the app's
external files dir), matching `com.scamshield.app.runtime.WhisperConfig`:

| File | Direction | Meaning |
| --- | --- | --- |
| `whisper_in.raw` | app → daemon | mono float32 LE PCM window |
| `whisper_in.ready` | app → daemon | request marker (presence ⇒ request ready) |
| `whisper_out.txt` | daemon → app | UTF-8 transcript |
| `whisper_out.ready` | daemon → app | result marker |
| `whisper_err.ready` | daemon → app | error marker |

Per request the daemon: clears stale `out.ready`/`err.ready` → reads
`whisper_in.raw` → mel → encoder → decode → detok → writes `whisper_out.txt`
**fully** → then creates `whisper_out.ready` (or `whisper_err.ready` on any
failure) → finally deletes `whisper_in.ready` then `whisper_in.raw`. It logs one
line per request to stderr (`[watch] <N> samples -> "<text>"`). Contexts stay
resident across requests.

## What it does

1. `dlopen` the backend (`libQnnHtp.so`) and `libQnnSystem.so`; resolve the
   `QnnInterface_getProviders` / `QnnSystemInterface_getProviders` vtables.
2. `logCreate` → `backendCreate` → `deviceCreate` (once, reused for both graphs).
3. For each `.bin`: read the file, `systemContextCreate` +
   `systemContextGetBinaryInfo` to enumerate the graph and its input/output
   `Qnn_Tensor_t` metadata, `contextCreateFromBinary`, then `graphRetrieve` to
   get the `Qnn_GraphHandle_t`. A name→tensor map is built and a zero-initialized
   client buffer is allocated and bound for every input/output tensor.
4. Load `input_features.raw` (fp16, `[1,80,3000]`, 480000 bytes) directly into the
   encoder's `input_features` buffer, run the encoder, and copy each cross-KV
   output into the matching decoder cross-KV input **by name**.
5. Run the greedy decode loop (forced English-transcribe prefix
   `[50258, 50259, 50359, 50363]`), feeding self-KV `*_out` back into `*_in` after
   each step (separate buffers, copied — never aliased), updating
   `attention_mask` / `position_ids` / `input_ids`, argmax over the 51865-wide
   fp16 logits, terminating on EOT (`50257`) or the decode budget.
6. Legacy mode prints the token IDs; `--pcm`/`--watch` detokenize to UTF-8 text.

In `--pcm`/`--watch` the front-end runs **before** step 4: PCM → log-Mel features
are written straight into the encoder's `input_features` buffer (no
`input_features.raw` file needed), and after step 5 the token IDs are detokenized
to text. The daemon re-seeds the decode state (attention mask, position, self-KV
caches) on every request so each window starts clean.

All graph I/O is populated and read **by tensor name** (never positional order),
so the per-layer interleaving of the KV tensors in the graph signature does not
matter.

## I/O contract (from the model `metadata.json`)

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `input_features` (enc in) | `[1,80,3000]` | fp16 |
| `k_cache_cross_{0..3}` (enc out / dec in) | `[6,1,64,1500]` | fp16 |
| `v_cache_cross_{0..3}` (enc out / dec in) | `[6,1,1500,64]` | fp16 |
| `input_ids` (dec in) | `[1,1]` | int32 |
| `position_ids` (dec in) | `[1]` | int32 |
| `attention_mask` (dec in) | `[1,1,1,200]` | fp16 |
| `k/v_cache_self_{i}_in` (dec in) | `[6,1,64,199]` / `[6,1,199,64]` | fp16 |
| `k/v_cache_self_{i}_out` (dec out) | same as `_in` | fp16 |
| `logits` (dec out) | `[1,51865,1,1]` | fp16 |

fp16 buffers use `_Float16` (native on aarch64 with ARMv8.2-A FP16).

## Build (host: macOS, target: Android arm64-v8a)

Requires the Android NDK (r26+). Two equivalent options:

### Option A — `build.sh` (direct clang++)

```bash
# Override QNN / NDK / ADB / RIG via env if your layout differs.
./build.sh
```

It compiles `whisper_qnn`, then prints the full `adb push` + on-device run
command block.

### Option B — CMake

```bash
NDK=/Users/sumakatabattuni/Library/Android/sdk/ndk/26.3.11579264
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DANDROID_STL=c++_static \
  -DQNN_SDK=/path/to/qairt/2.45.0.260326/include/QNN
cmake --build build-android -j
```

The QNN backend libraries are **not** linked at build time — they are `dlopen`'d
at runtime, exactly like the QNN SampleApp / `qnn-net-run`. Only `-ldl` (plus
`-lm`/`-llog`) is needed at link time.

## Run on-device (S25 Ultra, SM8750, Hexagon V79)

Push the runner, the QNN runtime libs, the V79 DSP skel, the two model bins, and
the input features into one rig dir, then point both `LD_LIBRARY_PATH` and
`ADSP_LIBRARY_PATH` at it:

```bash
RIG=/data/local/tmp/whisper_rig
adb shell "mkdir -p $RIG"
adb push whisper_qnn $RIG
adb push $QNN/lib/aarch64-android/libQnnHtp.so          $RIG
adb push $QNN/lib/aarch64-android/libQnnHtpPrepare.so   $RIG
adb push $QNN/lib/aarch64-android/libQnnHtpV79Stub.so   $RIG
adb push $QNN/lib/aarch64-android/libQnnSystem.so       $RIG
adb push $QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so $RIG
adb push encoder.bin decoder.bin vocab.bin mel_filters.bin $RIG

# Legacy: precomputed fp16 features -> token IDs
adb push input_features.raw $RIG
adb shell "cd $RIG && \
  export LD_LIBRARY_PATH=$RIG && \
  export ADSP_LIBRARY_PATH=$RIG && \
  chmod +x ./whisper_qnn && \
  ./whisper_qnn encoder.bin decoder.bin input_features.raw libQnnHtp.so"

# PCM -> text (front-end + detokenizer on-device)
adb push window.raw $RIG    # mono float32 LE PCM, any length
adb shell "cd $RIG && export LD_LIBRARY_PATH=$RIG && export ADSP_LIBRARY_PATH=$RIG && \
  ./whisper_qnn --pcm encoder.bin decoder.bin vocab.bin mel_filters.bin window.raw libQnnHtp.so"

# Resident daemon serving the app's file channel
CHAN=/sdcard/Android/data/com.scamshield.app/files
adb shell "cd $RIG && export LD_LIBRARY_PATH=$RIG && export ADSP_LIBRARY_PATH=$RIG && \
  ./whisper_qnn --watch encoder.bin decoder.bin vocab.bin mel_filters.bin $CHAN libQnnHtp.so"
```

Why both env vars: `LD_LIBRARY_PATH` lets the CPU-side `dlopen` find
`libQnnHtp*.so` / `libQnnSystem.so`; `ADSP_LIBRARY_PATH` lets the FastRPC/cDSP
loader find the Hexagon skel (`libQnnHtpV79Skel.so`). If graph finalize fails to
find the skel, broaden the search list:
`ADSP_LIBRARY_PATH="$RIG;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"`.

This out-of-process `/data/local/tmp` runner is the proven path that reaches the
NPU on a retail S25 (the in-app QNN delegate is blocked by SELinux from the
cDSP — see project memory `retail-selinux-npu-path`).

## Producing `input_features.raw` (legacy mode only)

The legacy positional mode expects the 80-bin log-Mel spectrogram for a 30 s
window, `[1, 80, 3000]`, as **raw little-endian fp16** (480000 bytes, no header).
`--pcm`/`--watch` don't need this file — they compute the features on-device from
raw PCM via the built-in mel front-end. To produce a reference for validation,
use `--dump-mel` and diff against the Python `WhisperFeatureExtractor`:

```bash
./whisper_qnn --dump-mel mel_filters.bin window.raw out.raw   # writes 480000 bytes fp16
```

## Exit codes

`0` success · `1` usage · `2` library load · `3` backend/device · `4` model load ·
`5` input/asset load (features/mel_filters/vocab/pcm) · `6` encoder/cross-KV ·
`7` decoder tensor lookup · `8` decode step / transcription. The `--watch` daemon
never returns `0` (it loops forever) — non-fatal per-request failures are
signaled to the app via `whisper_err.ready`, not the process exit code.
