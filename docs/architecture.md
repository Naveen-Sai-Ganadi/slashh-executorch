# Architecture

The complete on-device signal pipeline, every component, and every threshold. This is the
companion to [`stress-architecture.svg`](stress-architecture.svg).

![Architecture](stress-architecture.svg)

## Design principle

> One responsive, reliable audio signal drives the meter moment-to-moment; two **neural models on
> the NPU** (WavLM audio + Whisper→text) refine it through a fusion that only commits to "stressed"
> when the evidence is genuinely strong. Everything is on-device.

## End-to-end flow

```
Microphone (16 kHz mono)
  │  AudioCapture: ring buffer → emits a 3 s window (48000 samples) every 1 s hop
  ▼
VAD gate (RMS + zero-crossing-rate, adaptive noise floor)   ── unvoiced windows don't move the meter
  │
  ├──────────────── AUDIO SCORE ────────────────┐      ┌──────────── TEXT SCORE ────────────┐
  ▼                                              │      ▼                                     │
WavLmCoordinator (async, on its own thread)      │   WhisperBuffer (10 s rolling) + Coordinator│
  → NpuHelperScorer → file channel               │   → NpuWhisperTranscriber → file channel    │
  → npu_helper.sh (shell) → qnn-net-run           │   → whisper_qnn --watch (shell)             │
  → WavLM forward_2.bin on Hexagon NPU            │   → mel → encoder.bin → decoder loop → detok│
  → logit → sigmoid → EMA(α=0.5), throttle 2 s    │   → transcript text                         │
                                                  │   → TextFeatures (tokenize → L2 BoW, 2000)  │
  Energy (vad.lastRms → anchors 0.03/0.14)        │   → text_stress.pte  (Linear 2000→64→1 σ)  │
  is the FALLBACK audio leg when WavLM is stale   │   → text score ∈ [0,1]                      │
  → audio score ∈ [0,1]  ◀──────────────────────┘      └─────────────┬───────────────────────┘
                          ▼                                            ▼
                    FUSION   fusion.pte   Linear(2,1) → sigmoid   (symmetric, swap-augmented)
                          ▼   fires (>0.5) only when  audio + text ≥ 1.2  (both ≈ 0.6+)
                    fused stress ∈ [0,1]
                          ▼
                    EMA smoothing (α = 0.4)
                          ▼
                    Hysteresis latch  (enter ≥ 0.62  ·  release < 0.45)
                          ▼
                    STRESSED  →  CalmCue → ReliefNotifier (local heads-up + relief overlay)
```

## Components

### Capture & VAD
- **`AudioCapture`** — `AudioRecord` at 16 kHz mono PCM-16, a ring buffer that emits the most
  recent **3 s window (48000 samples) every 1 s hop**. Latest-wins: if scoring is slow, stale
  windows are dropped.
- **`Vad`** — gates on **short-term energy (RMS)** above an adaptive noise floor *and*
  **zero-crossing rate** in the speech band. Unvoiced/silent windows never reach the models and
  don't move the meter. Params: energy margin **6 dB**, ZCR **0.02–0.35**, absolute floor
  **3e-4 RMS**.

### Audio score (WavLM, NPU)
- **`WavLmCoordinator`** (`ai.slashh.audio`) scores the latest window **asynchronously** on its own
  thread. A WavLM forward reloads a 614 MB context (~1–3 s — far slower than the 1 s hop), so it
  *cannot* be the synchronous scorer. It is **throttled to one forward per 2 s** and the published
  score is **EMA-smoothed (α=0.5)** (each call scores a different window, so raw scores jump). Stale
  after **30 s**.
- **`NpuHelperScorer`** is the file-channel client to the shell-domain `npu_helper.sh`
  (see [NPU runtime](npu-runtime.md)).
- **Energy fallback** — `toStress(vad.lastRms)` mapped between calibrated anchors (**calm 0.03 →
  stress 0.14**) is the responsive audio signal used when WavLM is stale / its helper is down, and
  is what drives the meter between WavLM updates.

### Text score (Whisper → classifier, NPU + CPU)
- **`WhisperBuffer`** reconstructs a continuous **10 s** buffer from the 1 s hops.
- **`TranscriptionCoordinator`** transcribes on a **4 s cadence** (min 3 s of audio), off the audio
  path so it never blocks scoring. Transcripts are stale after **20 s**; a request times out after
  **8 s**. Whisper silence tokens (`[BLANK_AUDIO]`, etc.) are filtered out.
- **`NpuWhisperTranscriber`** is the file-channel client to `whisper_qnn --watch`.
- **`TextFeatures`** — the tokenizer + vectorizer (the Kotlin mirror of `model/text_features.py`):
  lowercase → replace every non-`a-z` with a space → unigrams → **L2-normalized bag-of-words over a
  fixed 2000-word vocabulary**.
- **`text_stress.pte`** — `Linear(2000→64) → ReLU → Linear(64→1) → sigmoid`, trained on the
  **Dreaddit** stress corpus (~70.9% test accuracy). Output = `text score ∈ [0,1]`. Runs on CPU via
  ExecuTorch XNNPACK (it's tiny).

### Fusion
- **`fusion.pte`** — a **single symmetric perceptron** `Linear(2,1) → sigmoid` over
  `[audio, text]`. Trained with **swap-augmentation** so the two inputs weigh equally (neither
  dominates), then its bias is calibrated so it crosses 0.5 only when **`audio + text ≥ 1.2`** (both
  ≈ 0.6+). Sample grid: `fuse(0.5,0.5)=0.14 · fuse(0.6,0.6)=0.50 · fuse(0.7,0.7)=0.86 ·
  fuse(0.8,0.8)=0.98`. See [Models](models.md) for why it's a perceptron and not the earlier MLP.
- The fusion's **audio leg** is the WavLM score when fresh, else the energy signal.

### Decision & relief
- **EMA** (`α=0.4`) smooths the fused/meter value so it's steady, not jittery.
- **Hysteresis latch** — enters "stressed" at **≥ 0.62**, releases below **0.45** (so it doesn't
  flicker at the boundary). The in-app **Sensitivity** slider shifts this live.
- **`CalmCue` / `ReliefNotifier`** — on a sustained latch, fire a high-priority **local**
  notification + an in-app relief overlay (breathing / sounds / colour-tap / etc.), rate-limited so
  it never nags.

### UI
- A React/Vite app in a WebView. The native side publishes state JSON (stress %, backend badge,
  **transcript**, **text/audio/fused** breakdown) via `window.updateAndroidState`; the
  **Settings → Live transcription** panel renders it for the active session (no history, nothing
  stored).

## All thresholds in one place

| Group | Parameter | Value |
|-------|-----------|-------|
| Capture | sample rate / window / hop | 16 kHz · 3 s (48000) · 1 s (16000) |
| Capture | log-mel | n_mels 64 · n_frames 301 (n_fft 400, hop 160) |
| VAD | energy margin · ZCR band · abs floor | 6 dB · 0.02–0.35 · 3e-4 RMS |
| Energy | calm / stress anchors | 0.03 / 0.14 |
| WavLM | throttle · EMA · freshness | 2 s · α=0.5 · 30 s |
| WavLM | memory guard | engage only if ≥ 1500 MB free |
| Whisper | buffer · cadence · min · freshness · timeout | 10 s · 4 s · 3 s · 20 s · 8 s |
| Text model | vocab · features | 2000 · L2 bag-of-words |
| Fusion | decision boundary | `audio + text ≥ 1.2` (`DECISION_BOUNDARY_SUM`) |
| Decision | EMA · latch enter / release | α=0.4 · 0.62 / 0.45 |

## Graceful degradation

| Situation | Behaviour |
|-----------|-----------|
| Both daemons up, RAM free | WavLM (NPU) audio + Whisper (NPU) text → fusion. |
| Low memory (< 1500 MB) | Memory guard skips WavLM → energy audio + text fusion. |
| Whisper daemon down | No transcript → audio-only (WavLM-or-energy), fusion idle. |
| No daemons at all | Energy signal only — the verified-reliable fallback. |

The in-app QNN delegate is **never** used on a retail S25 (it SIGSEGVs / is SELinux-blocked from the
cDSP); the NPU is always reached out-of-process. The CPU StressNet is also not used as a live
fallback (its native forward SIGSEGVs on-device) — energy is the dependable floor. See
[NPU runtime](npu-runtime.md) and [Research journey](research-journey.md).
