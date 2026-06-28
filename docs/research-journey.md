# Research journey — what we tried

The honest log of how Slashh got from "a tiny stress CNN" to "two transformers fused on the NPU."
The dead-ends are here too, because most of the engineering was in getting around them.

## The idea & the constraints

An **always-on, on-device voice stress detector** for the Qualcomm × Meta ExecuTorch Hackathon
(SF, June 27–28 2026; loaner **Galaxy S25 Ultra**, Snapdragon 8 Elite). We reframed "stress" as
**voice tension / arousal** — more defensible, and it maps cleanly to vocal features. The
non-negotiables: **everything on-device, no network, real-time**, and a story strong on the **40%
technical** bucket (NPU utilization, latency, energy).

## v1 — StressNet, and the RAVDESS saturation wall

The first model was **StressNet**: a tiny log-mel CNN (`[1,1,64,301] → 3×conv → GAP → linear →
sigmoid`, ~1.5–18 k params). Trained on **RAVDESS** (angry/fearful = stressed, neutral/calm = calm),
speaker-independent split, it hit **~96% balanced accuracy** offline and lowered cleanly:
**Qualcomm AI Hub showed 100% NPU (18/18 layers), ~0.06 ms/window** on the S25 Ultra. Great
headline numbers.

The wall: **on real on-device speech it saturated near 0.99 regardless of actual stress** — the
acted-vs-real domain gap. A model that says "stressed" to everything is useless live. So the live
meter fell back to **vocal energy (RMS)**, which is amplitude-invariant and dependable (silence →
low, loud → high). *Lesson learned early, and it stuck: energy is the reliable floor.*

## The branch sprawl

Three people, three app skeletons — we ended up with **three disjoint Android architectures** on
different branches (a React/Vite **WebView** app on `feature/ui-development`; a native **Canvas**
app on `sk/npu-binaries`; a **WavLM+NPU** minimal app on `ng/dev`), plus a `sk/demo-integration`
that curated the prettiest UI onto the WavLM base. Reconciling them (gradle wrappers, z-score vs
non-z-score log-mel, missing files) was its own small project. The current work builds on
`feature/ui-development`.

## WavLM on the NPU — and why it had to leave the app

We wanted a *real* neural audio model on the NPU, not just energy. **WavLM** (lowered to a QNN
context binary, `forward_2.bin`) ran on the **Hexagon V79** — proven live: NPU logit **−2.465** vs
eager **−2.555**, decision-agree.

But the **in-app QNN ExecuTorch delegate is SELinux-blocked** on a retail S25 (`sec_untrusted_app`
can't touch `/dev/fastrpc-cdsp`, error 4000). We also hit the real *other* blocker — the S25 Ultra's
**16 KB memory pages** rejecting non-aligned native libs. The breakthrough was to **stop trying to
run the NPU inside the app** and instead drive it from a **shell-domain helper** over a file
channel — the pattern that the whole runtime now rests on (see [NPU runtime](npu-runtime.md)).

## The always-on background monitor

`StressMonitorService` (foreground service, type microphone) made it always-on: it owns the mic,
hands each window to the shell helper over the file channel, and the app **mirrors** the live
reading onto the gauge. Energy still drove the latch; WavLM ran on the NPU per window as evidence.
This is the base the second feature plugged into.

## The second feature — Whisper + text + fusion

The leap from one signal to two: **also use *what* the person says, not just *how***.

1. **Port Whisper.** We had a working on-NPU Whisper-Tiny flow in a sibling project (`slashh-edge`,
   ScamShield). We ported it verbatim — same out-of-process file-channel pattern as WavLM, just
   different markers — into `ai.slashh.runtime` (`NpuWhisperTranscriber`, `WhisperBuffer`,
   `TranscriptionCoordinator`) plus the device rig (`tools/whisper/`).
2. **Train a text stress model.** A tiny **bag-of-words MLP** on the **Dreaddit** corpus
   (~70.9% test acc), exported to a small ExecuTorch `.pte`. **No LLM** — deterministic,
   sub-millisecond, on-device. Tokenizer is one source of truth shared by Python and Kotlin (parity
   tested).
3. **Fuse.** A small perceptron over `[audio, text]` → one confident stress probability.

On-device, the text model discriminated cleanly — *"I was a little scared"* → 0.42, *"overwhelmed
and anxious"* → 0.93, casual chatter → low.

## The four hard problems (and the fixes)

This is where the real time went.

### 1. The native SIGSEGV that "disabled the monitor"

The git history had a commit literally titled *"Temporarily disable StressMonitorService due to
native crashes."* Root cause, finally pinned: **a QNN-delegated `.pte` loaded by a CPU-only
ExecuTorch runtime crashes uncatchably** (a `try/catch` can't catch a native SIGSEGV). The CPU
StressNet forward also SIGSEGVs on this device. Fix: gate the QNN `.pte` on `qnnRuntimeAvailable()`,
and make the monitor's fallback the **energy** signal — never the in-app CPU model. The app went
from crash-at-startup to rock-stable.

### 2. The 614 MB out-of-memory spiral

WavLM's `qnn-net-run` rig **reloads a 614 MB context every call** (it isn't resident like Whisper's
`--watch`). With the device at ~790 MB free, driving WavLM actively meant every call OOM'd and the
**lowmemorykiller reaped the app** — and silently took both NPU daemons with it. We chased this hard
before realizing it was a genuine resource wall. The fixes, layered:
- A **reboot** frees memory to ~5–7 GB.
- A **memory guard** (`≥ 1500 MB free`) that auto-skips WavLM when RAM is tight.
- A new **`WavLmCoordinator`** that scores WavLM **async** on its own thread (it's ~1–3 s/call, far
  slower than the 1 s hop), **throttled + EMA-smoothed**, so the meter never freezes and the churn
  never runs back-to-back.

This is what finally got **both** models on the NPU, stably, without freezing the UI.

### 3. The FUSE cross-uid read race

After a reinstall the app's uid changed, and the shell-written `whisper_out.txt` (mode 660) on the
`/sdcard` **FUSE** mount intermittently `EACCES`'d for the app — so transcriptions were produced on
the NPU but the app couldn't read them. Fixes: the helper `chmod 0666`s its outputs (+ a small relay
for the Whisper binary, which writes 660 and can't chmod itself), and the app **retries the read**.
Suddenly the live loop closed: *"I was like, what is this?"* → textScore 0.74, on the phone.

### 4. The fusion-calibration saga

This took three iterations to feel right:
- **v1 — audio-dominant MLP.** The first `Linear(2,16)→ReLU→Linear(16,1)` fusion learned to weight
  *audio* heavily (the cleaner training signal). Result: high text + high audio could still produce
  a *low* fused score — "it doesn't make sense."
- **v2 — symmetric perceptron.** We swapped to a **single `Linear(2,1)` perceptron** with
  **swap-augmented** training so audio and text weigh equally. Now both contributed... but it fired
  on *moderate* + *moderate*, so ordinary speech read as stressed too often.
- **v3 — conservative boundary.** We kept the symmetric perceptron but calibrated its bias so it
  only crosses 0.5 when **`audio + text ≥ 1.2`** (both ≈ 0.6+), and raised the latch to enter
  0.62 / release 0.45. Casual chatter stays calm; genuine, sustained stress (tense voice *and*
  anxious words) fires. One knob (`DECISION_BOUNDARY_SUM`) tunes it.

A related, deliberate call: WavLM is **noisy and 15-s-lagged** live (and per the v1 finding,
doesn't discriminate real speech well), so we feed the fusion the **responsive energy** signal as
the audio leg when WavLM is stale, and keep WavLM on the NPU as the neural evidence. Honest about
what's reliable.

## Where it landed

Both neural models — **WavLM (audio)** and **Whisper-Tiny (text)** — run on the **Hexagon NPU**,
out-of-process, simultaneously, on a retail S25, without freezing a 1-second meter. A symmetric,
conservatively-calibrated perceptron fuses them; energy is the dependable floor; and **nothing —
audio, transcript, or score — ever leaves the phone**. No network. No LLM.

## Lessons

- **Energy is the floor.** A simple, reliable signal beats a fancy model that doesn't generalize.
- **Get off the cDSP-in-app path early.** Out-of-process via a shell helper is *the* way to reach a
  retail S25's NPU.
- **Resident beats per-request.** Whisper's `--watch` daemon is cheap; WavLM's per-call 614 MB
  reload caused most of the pain. (A resident WavLM runner is the obvious next improvement.)
- **Calibrate the fusion to human intuition, not just the data.** Symmetric + a conservative
  boundary made it *feel* right.
- **A native SIGSEGV is uncatchable** — prevent the load, don't try to catch the crash.
