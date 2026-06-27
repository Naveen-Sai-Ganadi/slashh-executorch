# Slashh AI v2 — Redesign Brainstorm & Decision Log

**Date:** 2026-06-26
**Status:** BRAINSTORMING (not yet an approved spec — decisions still open)
**Hard gate:** No implementation/scaffolding/code until a design is presented AND the
user approves it. The terminal step of brainstorming is to write the final spec
(`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`) and then a plan. This file is
the *working record* of the discussion that leads there.

---

## 1. The pivot — what the user asked for (v2)

A fundamentally more ambitious product than v1's tiny synthetic-trained StressNet:

- **Target device:** Samsung Galaxy **S25** Snapdragon (not S24).
  - *Now:* validate on **cloud S25 via AI Hub** + **local Mac** (live voice).
  - *At the hackathon:* a **real physical S25** Snapdragon device. Keep `.pte` + runner
    drop-in ready for the device.
- **Model:** use **any open-source model** instead of the tiny custom StressNet.
- **Test live with the user's own voice**, locally.
- **Run on-device on the S25** → requires quantization.
- "**All sorts of optimizations.**"
- **Set benchmarks.**
- **Finetune on REAL data, NO synthetic data.**
- **Dataset preparation with a huge dataset.**
- Model must **adapt to the user's voice** (personalization) for "true stress heuristics."
- "**Consider all possibilities — we need to win this hackathon at any cost.**"

---

## 2. Locked decisions (settled in conversation)

| # | Decision | Notes |
|---|---|---|
| L1 | Target = S25 Snapdragon; validate now on cloud-S25-via-AI-Hub + Mac | physical device at event |
| L2 | Architecture: real audio → frozen open-source backbone → embedding → global head + per-user prototype blend → calm/stressed score | |
| L3 | Training compute = Mac M4 Pro (48 GB, MPS) | frozen backbone ⇒ no heavy backprop; Colab only for optional full fine-tune |
| L4 | Real corpora only, NO synthetic (RAVDESS + CREMA-D + TESS + SAVEE) | arousal labels via Russell's circumplex |
| L5 | Personalization = embedding prototypes (per-user calm + optional stressed centroids) | blended with global model |
| L6 | Decomposition: SP1 data → SP2 backbone+head → SP5 mic demo → SP3 personalization → SP4 on-device S25 → SP6 benchmarks | |

### Emotion → arousal mapping (Russell's circumplex)
- High-arousal → **stressed (1.0):** anger, fear, surprise, happy, disgust
- Low-arousal → **calm (0.0):** neutral, calm, sad, bored

---

## 3. AI Hub — what it is and is NOT (correction of a user misconception)

Qualcomm AI Hub (`qai_hub` 0.50.0) **does NOT train models and has NO GPUs.** It only
**compiles / quantizes / links / profiles / runs inference** on real Snapdragon hardware
in a cloud device farm. It consumes already-trained **TorchScript/ONNX** (not `.pte`).
Five Workbench tabs: COMPILE, QUANTIZE, LINK, PROFILE, INFERENCE. Training happens on the
Mac; AI Hub is deployment/measurement only.

---

## 4. Backbone recon — the key technical investigation

### 4.1 The zoo (`qai_hub_models`) contains 195 models. Only 3 are usable audio encoders.

Everything else is vision, LLMs, or TTS. The audio encoders:

| Model | What it is | Embedding | Inductive bias |
|---|---|---|---|
| **whisper_tiny** (encoder) | ASR transformer encoder | 384-d | content / phonetic |
| **zipformer** | streaming ASR encoder (k2, **Chinese-English mixed** ckpt) | frame-level | content / phonetic |
| **yamnet** | AudioSet event classifier (MobileNetV1) | 1024-d | **timbre / acoustic-event** |

**Traps verified:**
- `xlsr` *sounds* like wav2vec2-XLSR but is an **image super-resolution** model in this zoo.
- `melotts_*`, `pipertts_*` are **TTS** (text→audio) — useless as extractors.
- There is **NO wav2vec2 / HuBERT / WavLM / dedicated SER** model in the zoo.

**Correction logged:** earlier in the conversation I claimed "Whisper encoders are the
ONLY speech backbone." That was wrong — **yamnet and zipformer are also on-device-ready
audio encoders.** yamnet is a genuine A/B contender (see 4.3).

### 4.2 Why Whisper-tiny among the three on-device-capable encoders
- **vs zipformer:** off-label Chinese-English ASR checkpoint; RNN-T (encoder+decoder+joiner)
  is messy to separate. Higher risk, no upside.
- **vs yamnet:** genuinely competitive — kept as an A/B arm.
- **Whisper-tiny's case:** smallest speech transformer with a *separable* encoder
  (`@CollectionModel.add_component(HfWhisperEncoder, "encoder")`), multilingual/English
  coverage, a **quantized Snapdragon recipe already in the zoo**
  (`whisper_small_quantized` proves the recipe path), and documented SER transfer.
  Whisper-tiny `d_model = 384`.

### 4.3 The catch with Whisper for stress, and why yamnet matters
Whisper's encoder is trained for **transcription** → deliberately **invariant to how you
sound** (stressed/calm/angry) so it recovers the same words. That affect-invariance is the
**wrong inductive bias for paralinguistics.** yamnet has the opposite bias: trained on
AudioSet events (classes include *shout, sigh, groan, breathing, crying*), its embedding
encodes vocal **timbre/texture** — closer to arousal. So for *stress specifically*, yamnet
may beat Whisper despite being the "simpler" model. ⇒ real A/B arm, not a footnote.

---

## 5. Are there better open-source models? Yes — on accuracy, none are in the zoo

Ranked best-first for speech-emotion accuracy (all **outside** the zoo, no Snapdragon recipe):

1. **emotion2vec** — self-supervised, pre-trained *specifically for emotion*. Tops SER benchmarks.
2. **WavLM-large** — SUPERB leader for paralinguistic tasks.
3. **wav2vec2 / HuBERT emotion fine-tunes** — e.g. **audeering wav2vec2-large** (MSP-Podcast)
   outputs **arousal / dominance / valence directly** = literally a pretrained arousal regressor
   (our exact target variable).

All beat a frozen Whisper-tiny encoder. The **only** blocker is deployment: zero Snapdragon
recipes ⇒ hand-rolled NPU export, risky the night before a demo.

### 5.1 The insight that dissolves the accuracy-vs-deployability tradeoff: **distillation**
- Run a **SOTA model off-device as a teacher** (emotion2vec / audeering wav2vec2-arousal) on
  the **Mac, never the phone**, to produce soft labels / arousal targets on real data.
- **Distill into a tiny on-device student** — Whisper-tiny encoder + head, *or even keep the
  1,549-param StressNet* (already 100% NPU, ~71 µs).
- Teacher never runs on Snapdragon; student inherits SOTA accuracy while staying NPU-resident.

This adds a **distillation track** to the v2 plan that was previously under-scoped, and is the
strongest "win at any cost" move.

---

## 6. Feature-selection menu (what can feed the backbone/head)

> Note: v1 does **no explicit feature selection** — `model/features.py` hands the CNN a raw
> 64×301 log-mel and learns features end-to-end. No CMVN, no deltas, no prosody. In v2,
> "feature selection" = choosing the representation that feeds the backbone/head.

**1. Time–frequency representations**
- Log-mel spectrogram (current, 64×301, NPU-proven) · MFCC (+Δ, ΔΔ) · linear STFT · mel
  filterbank energies · CQT · gammatone filterbank.

**2. Prosodic / eGeMAPS (strongest *explicit* arousal signal)**
- F0 (mean/std/range/slope) · energy/intensity (RMS, loudness) · **jitter** · **shimmer**
  · **HNR** · voicing probability · speaking rate / pause ratio · formants F1–F3 · spectral
  tilt (H1–H2, H1–A3) · cepstral peak prominence. Packaged set = **eGeMAPS (88 descriptors)**.

**3. Spectral-shape descriptors**
- centroid, spread, flux, rolloff, slope, entropy, flatness · **alpha ratio**, **Hammarberg
  index** · zero-crossing rate (already computed in `Vad.kt` → free).

**4. Learned embeddings**
- Whisper-tiny encoder (384) ✅ recipe · Whisper-base (512) ✅ · yamnet (1024) ✅ · wav2vec2/
  HuBERT/WavLM/emotion2vec ❌ no recipe (teacher-only).

**5. Normalization / augmentation transforms (current pipeline has NONE — known gap)**
- **CMVN** (per-utterance/per-speaker) · **PCEN** · loudness/level norm (our gain-robustness
  gap) · Δ/ΔΔ stacking · speaker norm (VTLN — feeds personalization).

### 6.1 Proposed feature A/B for SP2 (revised to include yamnet)
**Whisper-384 vs yamnet-1024 vs eGeMAPS-88 vs fusion** — all with **CMVN/level-norm added**
(closes the known "log-mel has no level normalization" gain gap), with the **off-device
teacher supplying arousal labels for every arm.**

---

## 7. Open decisions (need user confirmation before writing the spec)

| # | Decision | Recommendation | Why it's open |
|---|---|---|---|
| O1 | Backbone-to-deploy | Whisper-tiny **+ yamnet** as a 2-arm on-device A/B; pull accuracy ceiling from an **off-device teacher via distillation** | recon forces "no wav2vec2 on-device"; yamnet vs Whisper bias is a real question |
| O2 | Which sub-project to spec first | **SP1 dataset prep** (everything depends on it; no audio on disk, `realdata.py` only scans) | SP5 mic demo is defensible if user wants fastest "hear my own voice" payoff |
| O3 | Fold 3/4-arm feature A/B + CMVN into SP2 | **Yes, staged** — get Whisper-384 working end-to-end first, then run other arms on the same harness | widens SP2 scope; could defer as later optimization |
| O4 | **NEW:** add the teacher→student distillation track | **Yes** — best "win at any cost" lever | introduced after the zoo recon; not yet user-confirmed |

---

## 8. Sub-project decomposition (build order)

1. **SP1 — Dataset prep** (greenfield). Downloader for RAVDESS + CREMA-D + TESS + SAVEE;
   arousal labelling; held-out split. *Unblocker for everything.* `realdata.py` currently
   only *scans* local paths — no downloader, no audio on disk.
2. **SP2 — Backbone + head** (the model). Frozen backbone → embedding → tiny head. Hosts the
   feature A/B (§6.1) and the distillation track (§5.1).
3. **SP5 — Local mic demo.** Live Mac mic → score; "hear my own voice."
4. **SP3 — Personalization.** Per-user calm/stressed embedding prototypes blended with global head.
5. **SP4 — On-device S25.** Quantize → compile → profile → inference on Snapdragon (AI Hub now, physical device at event).
6. **SP6 — Benchmarks.** XNNPACK-CPU vs QNN-NPU latency, model size, accuracy; reproducible quickstart.

Build order rationale: SP1 → SP2 → SP5 → SP3 → SP4 → SP6.

---

## 9. Host hardware (training feasibility)
Apple **M4 Pro, 48 GB RAM, 12 cores**, PyTorch 2.11.0, **MPS available**. Sufficient for
frozen-backbone training: extract embeddings once (cacheable on MPS), train tiny head
(seconds, CPU), personalize via centroids (no backprop). Colab only needed for an optional
full backbone fine-tune. AI Hub never does a backward pass.

---

## 10. Security / process constraints still in force
- Live Qualcomm AI Hub API token: **never commit or echo.** Read only by parsing `.env` in
  Python → write `~/.qai_hub/client.ini`. `.gitignore` excludes `client.ini`, `*.token`, `.env`.
  Token key `API_TOKEN` in `.env`. Rotate again after the event (June 27-28, 2026).
- AI Hub job submission spends **live credits** → **human-gated**.
- Secret-scan before any commit. `.claude/hooks/git-guard.sh` + `lib.sh` are **immutable by policy**.
- Pushes to `main` blocked without a fresh green sentinel; non-main branch pushes allowed.
- NEVER stage `android/`, `Vad.kt`, `run_jvm_tests.sh` in host-only commits.
- Entering passwords / authenticating is **prohibited**.

---

## 11. Branch context (relevant to where v2 should be based)
- `dev` now fully contains `feature/night-vad-fix-and-jvm-tests` (night-vad is an ancestor) and
  is **3 commits ahead** via contributor **Suma Katabattuni** (Android build fix, premium dark
  UI redesign, PR #3 merge). `dev` is the **live integration branch**.
- ⇒ v2 work should branch off **`dev`**, not `main`, to stay aligned with the live Android app.

---

## 12. Next step
Confirm O1–O4 (§7). Then: write the final spec to `docs/superpowers/specs/`, user reviews it,
and only then move to the implementation plan (writing-plans). No code until that approval.
