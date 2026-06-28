# NPU runtime — two models on the Hexagon, out-of-process

How Slashh runs **WavLM** (audio) *and* **Whisper-Tiny** (ASR) on the Snapdragon Hexagon NPU at the
same time on a **retail** Galaxy S25, why it has to be out-of-process, and every gotcha we hit
bringing both up.

## Why out-of-process (the SELinux story)

On a retail (locked) Galaxy S25 the app process runs in the `sec_untrusted_app` SELinux domain,
which is **blocked from `/dev/fastrpc-cdsp`** — so an **in-app QNN ExecuTorch delegate fails to
reach the NPU** (error 4000, or a hard native crash). The **shell domain** (uid 2000, what `adb
shell` runs as) *does* have cDSP access.

So each model runs as a **shell-domain helper** that drives the QNN context binary on the Hexagon,
and the app exchanges data with it over a **file channel** in the app's own external-files dir
(`/sdcard/Android/data/ai.slashh/files`). The app and the shell helper share that dir via the
`ext_data_rw` group. This needs **no extra permission and no `INTERNET`** — audio never leaves the
device.

```
App (untrusted_app)                     file channel                Shell domain (uid 2000)
  NpuHelperScorer / NpuWhisperTranscriber  ───►  npu_in.raw / whisper_in.raw  ───►  npu_helper.sh / whisper_qnn --watch
                                                  *.ready markers                     → qnn-net-run / QNN C-API on Hexagon NPU
  reads ◄──────────────────────────────────────  npu_out.raw / whisper_out.txt  ◄──  writes
```

Two independent rigs share the one channel dir; their markers never collide:

| Model | Helper / runner | Rig dir | Channel markers |
|-------|-----------------|---------|-----------------|
| WavLM (audio) | `npu_helper.sh` → `qnn-net-run` | `/data/local/tmp/qnntest` | `npu_in.*` / `npu_out.*` / `npu_err.*` |
| Whisper (text) | `whisper_qnn --watch` | `/data/local/tmp/whisper_rig` | `whisper_in.*` / `whisper_out.*` / `whisper_err.*` |

A key difference: **`whisper_qnn --watch` is resident** (loads the encoder/decoder context **once**
and serves many requests, ~150 ms each), whereas **`npu_helper.sh` runs `qnn-net-run` per request**,
which **reloads its 614 MB context every call** (~1–3 s). That single fact drives most of the
gotchas below.

## Obtaining the QNN context binaries

The large binaries are **not vendored** in the repo (≈ 0.7 GB total). What *is* committed:
`tools/whisper/` (the `whisper_qnn` runner binary, `vocab.bin`, `mel_filters.bin`, helper script,
source + `CMakeLists`) and `npu_helper.sh` (the WavLM helper).

| Artifact | Size | Source |
|----------|------|--------|
| Whisper `encoder.bin` / `decoder.bin` | 19 MB / 97 MB | **Qualcomm AI Hub** Whisper-Tiny QNN context-binary export for `qualcomm-snapdragon-8-elite` (float). Place under `tools/whisper/whisper_tiny_qnn/`. |
| WavLM `forward_2.bin` | 614 MB | WavLM lowered to a QNN context binary (AI Hub / QAIRT). See the v2 WavLM-on-NPU proof under [`v2/`](v2/). |
| QAIRT runtime `.so`s | — | **QAIRT 2.45** for Whisper, **QAIRT 2.37** for WavLM (`libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV79Skel.so`, …). The Community editions auto-download without a Qualcomm login. |

> **Version skew is fatal.** A context binary compiled against QAIRT *X* must run against the same
> `libQnnHtp*.so` / `qnn-net-run` version *X* on-device, or graph finalize fails. Keep Whisper on
> 2.45 and WavLM on 2.37 — they live in separate rig dirs, so they don't conflict.

`tools/whisper/npu_helper_whisper.sh` stages the rig and (with `QNN=<sdk>` set) pushes the QAIRT
libs; the WavLM rig is staged per the v2 notes.

## Bringing both daemons up (after any boot)

Both daemons die on reboot. From the host (with `$ADB` set):

```bash
# Whisper (resident) — stages rig + starts daemon + the chmod relay, points at ai.slashh's channel
ADB="$ADB" ./tools/whisper/npu_helper_whisper.sh start

# WavLM (per-request qnn-net-run)
$ADB shell "cd /data/local/tmp/qnntest && nohup sh npu_helper.sh > npu_helper.log 2>&1 < /dev/null &"

# warm each once (first call loads the cold context), then (re)launch the app
$ADB shell "am force-stop ai.slashh; am start -n ai.slashh/.MainActivity"
```

The app's monitor probes the WavLM helper for up to **30 s** at startup (the cold 614 MB load can
take ~15 s). If it answers, the audio leg routes through WavLM; otherwise it falls back to energy.

> **The launch shell appears to hang.** `nohup … &` over `adb shell` doesn't always detach cleanly
> (the QNN process keeps the shell's fds open). The daemon **is** running — verify in a *separate*
> command: `adb shell "ps -A -o ARGS | grep 'whisper_qnn --watch'"`.

## Gotcha 1 — memory (the 614 MB reload)

WavLM's `qnn-net-run` reloads its 614 MB context **per request**. When the device sat at ~790 MB
free, every WavLM call pushed it over the edge and the **lowmemorykiller reaped the app**. Two fixes:

1. **A reboot** frees memory to ~5–7 GB; both daemons + the app then coexist comfortably.
2. **`StressMonitorService.buildPipeline` has a memory guard**: it only engages WavLM when
   `MemAvailable ≥ 1500 MB` (read from `/proc/meminfo`); below that it runs energy + text fusion.
   Self-adapting — WavLM re-engages once memory frees.
3. **`WavLmCoordinator` is async + throttled** (2 s) so the alloc/free churn never runs back-to-back.

## Gotcha 2 — the FUSE cross-uid read

The channel is a `/sdcard` **FUSE** mount. The shell helper writes `npu_out.raw` / `whisper_out.txt`
as mode **660**; the app (a *different* uid — and the uid changes when you reinstall) reads them. A
freshly-written cross-uid file can briefly `EACCES` for the app while FUSE propagates. Fixes (all
shipped):

- The WavLM helper **`chmod 0666`s its output**; the Whisper helper script starts a tiny **chmod
  relay** (`whisper_chmod_relay`) that keeps `whisper_out.*` world-readable (the `whisper_qnn`
  binary itself writes 660 and can't chmod from inside).
- The app **retries the read** (`NpuHelperScorer.readLogit`, `NpuWhisperTranscriber` — retry the
  open for ~1 s) so a transient race never drops a result.

## Gotcha 3 — the pre-existing in-app native crash

The "disable StressMonitorService" history (commit `76ca2b60`) was a **native SIGSEGV**: a
QNN-delegated `.pte` loaded by the **CPU-only** ExecuTorch runtime crashes uncatchably, and so does
the CPU StressNet forward on this device. Fixes:

- `MainActivity.loadClassifier` only tries the QNN `.pte` when `qnnRuntimeAvailable()` (a QNN
  backend native lib is bundled) — otherwise it loads the safe XNNPACK `.pte`.
- The monitor never wires the crashing CPU StressNet; it uses the **energy** fallback. The NPU is
  always reached **out-of-process**, never via the in-app delegate.

## Gotcha 4 — S25 Ultra 16 KB pages

The S25 Ultra (Android 15) uses **16 KB memory pages**; native libs must be 16 KB-aligned or they
won't load (`ELF alignment check failed`). The shipped libs are aligned (NDK r27c + injected
`-Wl,-z,max-page-size=16384`). This — not SELinux — was the real blocker for the *in-app* QNN AAR
path; we sidestep it entirely by running the NPU out-of-process.

## Health-check cheatsheet

```bash
$ADB shell "ps -A -o ARGS | grep -E 'whisper_qnn --watch|npu_helper.sh' | grep -v grep"   # daemons alive?
$ADB shell "grep MemAvailable /proc/meminfo"                                               # RAM headroom
./tools/whisper/npu_helper_whisper.sh status                                               # whisper log tail
$ADB logcat -d | grep ' Slashh' | grep -E 'WavLM on NPU|wavlm\[NPU\]|whisper\[|skipping WavLM'
```
