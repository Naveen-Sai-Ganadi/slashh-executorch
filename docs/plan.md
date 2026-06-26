# Slashh AI — ExecuTorch Hackathon Build Plan

**Project:** Real-time, on-device voice stress detection + instant calming, running fully on Snapdragon (Galaxy S25 Ultra). Data never leaves the phone.
**Team:** Naveen Ganadi · Suma Katabattuni · Rohit Somisetty
**Event:** Qualcomm × Meta ExecuTorch Hackathon — on-site San Francisco, June 27–28, 2026. 5-minute live demo.
**Stack mandate:** PyTorch → ExecuTorch, runs locally on Snapdragon, public GitHub repo, OSS license.

---

## 0. How we win 100/100 — the rubric IS the spec

Every hour of this build should be spent buying points on this scorecard. Memorize these weights; they decide everything below.

| # | Criterion | Weight | What judges actually look for | Our proof |
|---|-----------|:---:|-------------------------------|-----------|
| 1 | **Technical Implementation** | **40%** | NPU utilization, latency/perf, energy efficiency | Profile-job screenshot showing **NPU compute unit active**, measured per-window latency, INT8 quantized model, VAD duty-cycling |
| 2 | Application Use Case & Innovation | 25% | Problem-solving, creativity/uniqueness, UX | Closed **detect→calm loop** (not just a classifier), clean live stress meter UX |
| 3 | Local Processing & Privacy | 15% | On-device execution, privacy/security | **Airplane-mode live demo**, raw audio never persisted, zero network calls |
| 4 | Deployment & Accessibility | 10% | Ease of install/use | One APK, copy-paste README, runs first try |
| 5 | Presentation & Documentation | 10% | Demo clarity, code quality/docs | Rehearsed 5-min script, documented repo, backup video |

**The single most important sentence in this plan:** 40% of the score is *technical evidence*, and that evidence (NPU usage, latency, memory) is produced by **Qualcomm AI Hub Workbench profile jobs** + on-device measurement. If we have screenshots proving the model runs on the **Hexagon NPU** with real numbers, we own the biggest block on the board. Everything else is about not dropping the easy points.

**Priority legend used throughout:** `[CRITICAL]` = directly buys the 40%/privacy points, do first · `[IMPORTANT]` = required to ship · `[SUPPORTING]` = polish/insurance.

---

## 1. Project overview and goal

**What we're building:** "Slashh" — an Android app that listens to the user's voice in short rolling windows, detects rising stress on-device in real time, and the moment stress crosses a threshold, immediately launches a calming intervention (guided breathing + calming audio/haptics). No audio or result ever leaves the phone.

**Who it's for:** Anyone with stress/anxiety who wants a private, always-available, zero-trust check-in — people who would never send their voice to a cloud server.

**Core problem:** Stress detection tools today are either cloud-based (privacy nightmare for the most sensitive signal a person has — their voice) or passive trackers that *tell* you you're stressed but don't *do* anything about it.

**"Done" looks like:** On a Galaxy S25 Ultra in airplane mode, a judge speaks for ~15 seconds; the on-screen meter rises into "stressed"; within ~1 second the app auto-triggers a calming flow — and we can show a profile screenshot proving inference ran on the NPU.

**Single most important success criterion:** *Real-time stress inference running on the Snapdragon NPU, demonstrated live and offline, with measured latency.* When any tradeoff comes up, protect this.

---

## 2. Scope and non-goals

**In scope (v1, demo-critical):**
- Mic capture → VAD → stress/calm classification on-device via ExecuTorch on NPU.
- Live stress meter UI + automatic calming intervention on threshold crossing.
- Fully offline operation; profiling evidence of NPU + latency + memory.

**Explicit non-goals (do NOT build these — they burn on-site hours for zero rubric points):**
- ❌ No clinical/diagnostic claims. This is "elevated-arousal likelihood," not a medical device.
- ❌ No cloud, accounts, login, or backend.
- ❌ No multi-modal fusion (HRV, camera PPG, keystroke) in v1 — mention as future work only.
- ❌ No speech-to-text / no understanding *what* is said — we only model *how* it's said (also a privacy win).
- ❌ No personalization/baseline-learning pipeline in v1 (nice future-work talking point).
- ❌ No app-store polish, settings screens, or onboarding beyond the demo path.

---

## 3. Tech stack and constraints

**Languages/frameworks:**
- Model: **PyTorch** (training/export) → **ExecuTorch** (`.pte`) for on-device inference.
- App: **Android (Kotlin)**, Android Studio, min SDK ~26+, target S25 Ultra (Android 15 / One UI 7).
- DSP/feature extraction: log-mel spectrogram (compute on-device; torchaudio offline, custom Kotlin/C++ on-device).

**On-device runtime path (this is the heart of the 40%):**
- PyTorch model → **ExecuTorch export**. Start with the **XNNPACK backend** (CPU) to get end-to-end working fast, then switch to the **Qualcomm AI Engine Direct (QNN) backend** to push inference onto the **Hexagon NPU**.
- Use **Qualcomm AI Hub Workbench** (`qai-hub`, **installed & authenticated** — see §8.1) to compile, **quantize to INT8**, run inference, and **profile** on the hosted **Samsung Galaxy S25 Ultra** (Snapdragon 8 Elite for Galaxy — the exact loaner silicon, confirmed available in §8.3) — this generates the NPU/latency/memory evidence for our slides *before and during* the event.

**Hard constraints:**
- Majority must run **locally on device** (hybrid allowed but local must dominate — we go 100% local, which also maxes the 15% privacy block).
- Must demonstrably use **PyTorch + ExecuTorch + Snapdragon** (non-negotiable for criterion #1).
- Real-time budget: target **< 300 ms** inference per audio window on NPU (aim lower; lower = better latency score).
- OSS license required (use **MIT** or **Apache-2.0**); all dependencies must be OSS.
- 2-day on-site build → scope must fit. Pre-event prep is mandatory (see §14).

---

## 4. Architecture and project structure

**Data flow (closed loop):**
```
Mic (16kHz mono)
  → Ring buffer (3–5s rolling window, hop ~1s)
  → VAD gate  ── silence? ──▶ skip (saves energy)  [energy-efficiency point]
  → Log-mel spectrogram features
  → ExecuTorch model on NPU (QNN backend) → stress score [0..1]
  → Smoothing (EMA over last N windows) → stress LEVEL
  → UI: live meter
  → Threshold crossed? → Calming intervention (breathing + audio + haptics)
```

**Proposed repo structure:**
```
slashh-ai/
├── README.md                # team, description, setup, run, license (criterion #5 + #4)
├── LICENSE                  # MIT/Apache-2.0
├── model/
│   ├── train.py             # train/fine-tune stress classifier (PyTorch)
│   ├── export_executorch.py # PyTorch → .pte (XNNPACK then QNN)
│   ├── aihub_profile.py     # qai-hub compile/quantize/profile → evidence
│   ├── features.py          # log-mel extraction (parity with on-device)
│   └── eval.py              # offline accuracy/eval harness (criterion #1/#5)
├── android/
│   └── app/…                # Kotlin app: audio, VAD, ExecuTorch runner, UI, intervention
├── assets/
│   └── stress_model.pte     # exported model bundled in app
├── benchmarks/
│   └── results.md           # latency, NPU%, memory, energy notes + screenshots
├── docs/
│   ├── architecture.md
│   └── demo_script.md       # the 5-minute runbook
└── tests/
```

**Reuse, don't reinvent:** the hackathon explicitly provides a **Whisper Transcription** sample app and a participant **Voice Stress Detection Model** (ShieldHer) as fair starting points — clone the app scaffolding patterns and study the stress model rather than building audio plumbing from zero.

---

## 5. Data model and schemas

This is an audio app, not a CRUD app — keep state minimal and **on-device only**.

- `AudioWindow`: float[] PCM, 16kHz mono, length = window seconds. **Never persisted.** Lives in a ring buffer, overwritten continuously.
- `StressScore`: `{ timestamp, raw: float[0..1], smoothed: float[0..1], level: enum{CALM, RISING, STRESSED} }`. In-memory; optionally a rolling **session summary** (counts, peak) held in memory and discarded on close.
- `InterventionEvent`: `{ triggered_at, trigger_level, type: BREATHING }`. In-memory only.
- **Persistence rule (privacy):** if we store *anything* for a session-history feature, store only derived scores (never audio, never features that could reconstruct audio), in local app storage (Room/SQLite or a local file), with a visible "clear" action. Default to storing nothing.

**Validation rules:** clamp scores to [0,1]; require ≥1 voiced window before showing a level; ignore windows failing VAD.

---

## 6. Feature breakdown / task list (vertical slices, ordered)

Build one working end-to-end slice first, then deepen. Each task: purpose · output · depends-on.

1. `[CRITICAL]` **Model selection + offline classifier.** Pick/fine-tune a small audio emotion/stress model (see §13 datasets). Output: PyTorch model that classifies calm vs stressed from a log-mel window. *Do this BEFORE the event.*
2. `[CRITICAL]` **ExecuTorch export (XNNPACK).** Convert to `.pte`, verify identical output to PyTorch on CPU. Output: working `.pte`. Depends: 1.
3. `[CRITICAL]` **AI Hub profile + INT8 quantize + QNN/NPU.** Run compile/quantize/profile on S25 via Workbench; confirm **NPU compute unit** and capture latency/memory. Output: NPU-ready model + evidence screenshots. Depends: 2.
4. `[IMPORTANT]` **Android audio pipeline.** Mic capture → ring buffer → VAD → log-mel (parity with `features.py`). Output: live feature windows on device. Independent of 1–3, build in parallel.
5. `[CRITICAL]` **On-device ExecuTorch inference.** Load `.pte`, run per window on NPU, output stress score. Output: live scores on device. Depends: 3, 4.
6. `[IMPORTANT]` **Stress meter UI + smoothing.** EMA smoothing → CALM/RISING/STRESSED meter. Output: visible real-time meter. Depends: 5.
7. `[IMPORTANT]` **Calming intervention.** On threshold cross, auto-launch guided breathing (animation + calming tone + haptics). Output: the "calms you instantly" payoff. Depends: 6.
8. `[CRITICAL]` **Offline/airplane-mode hardening.** Guarantee zero network; verify in airplane mode. Output: privacy proof. Depends: 5.
9. `[IMPORTANT]` **Benchmarks + energy notes.** Record latency, NPU%, memory; argue energy efficiency. Output: `benchmarks/results.md`. Depends: 3, 5.
10. `[SUPPORTING]` **README, license, demo video, slides, rehearsal.** Depends: all.

**Minimum viable demo (if time collapses):** tasks 1→2→4→5→6→8. A live, offline, on-device meter that moves with stress — even on XNNPACK CPU — is a working demo. NPU (task 3) is the points multiplier; treat it as must-have but have the CPU path as the safety net.

---

## 7. API / interface contracts (module boundaries)

- `features.extract(pcm: float[]) -> melspec: float[][]` — deterministic; must match on-device implementation bit-for-bit-ish (validate parity, criterion #1 correctness).
- `StressModel.infer(melspec) -> {score: float[0..1]}` — single source of truth for the classifier; same I/O shape in PyTorch, ExecuTorch, and on-device.
- `VAD.isVoiced(window) -> bool` — gate before inference (energy point).
- `Intervention.trigger(level) -> Unit` — idempotent; don't re-trigger while already calming.
- **External SDKs:** `qai-hub` (compile/quantize/inference/profile jobs, no secrets in repo — API token via local config only); ExecuTorch runtime (QNN backend) in the Android app. No external network APIs at runtime — that's the whole point.

---

## 8. Environment, setup, and configuration

- **Python env (model):** `pip install 'qai-hub[torch]' executorch torchaudio`. Authenticate AI Hub: `qai-hub configure --api_token <TOKEN>` (token from Workbench account page; **never commit it** — document the env var name in README, human supplies the value). Verify with `qai-hub list-devices`.
- **Android:** Android Studio (latest), ExecuTorch Android runtime + QNN backend libraries, Gradle sync, USB debugging on the S25 Ultra (loaner — sign the device agreement at check-in).
- **Config/env vars (names only, no secrets):** `QAI_HUB_API_TOKEN`, `STRESS_THRESHOLD` (default e.g. 0.6), `WINDOW_SECONDS`, `HOP_SECONDS`.
- **Run locally:** documented one-liners in README — install deps, build APK, install to device, launch. Must work on a fresh machine following only the README (criterion #4).

### 8.1 Verified local environment (installed & authenticated — as of June 25, 2026, T-2 days)

Phase-0 AI Hub toolchain is **live and authenticated** on the dev machine. Reproduce exactly:

```sh
uv venv --python 3.11                 # Python 3.11.15
source .venv/bin/activate
uv pip install qai_hub qai_hub_models torch torchaudio executorch
qai-hub configure --api_token <QAI_HUB_API_TOKEN>   # token NOT stored in repo
qai-hub list-devices                  # smoke test — confirms auth + farm access ✅
```

**Pinned versions confirmed installed** (lock these in `model/requirements.txt`; export/quantization behavior is version-sensitive):

| Package | Version | Role |
|---|---|---|
| `executorch` | **1.2.0** | PyTorch → `.pte`, on-device runtime |
| `torch` / `torchaudio` / `torchvision` | **2.11.0 / 2.11.0 / 0.26.0** | model + log-mel features |
| `qai-hub` | **0.50.0** | compile / quantize / **profile** jobs (the 40% evidence engine) |
| `qai-hub-models` (`-cli`) | **0.56.0** | model zoo + ready ExecuTorch export/profile recipes (see §8.2) |
| `torchao` | **0.17.0** | INT8 quantization |
| `coremltools` / `onnx` / `onnxruntime` | 9.0 / 1.18.0 / 1.22.1 | peripheral backends (not the Snapdragon path) |

**Auth/config facts:** `qai-hub configure` writes `~/.qai_hub/client.ini` (`api_url = https://workbench.aihub.qualcomm.com`). That file is **outside the repo** and holds the token — keep it there; never copy it into the project. The repo `.gitignore` already excludes `.venv/`.

> **🔐 Security:** the API token was entered in plaintext on the shell. Treat it as exposed — **rotate it in Workbench after the event** (sooner if convenient). It must live only in `~/.qai_hub/client.ini` and as the `QAI_HUB_API_TOKEN` env var — never in git, plan.md, README, or slides.

### 8.2 Head start — `qai-hub-models` (don't hand-roll the export recipe)

`qai_hub_models` ships pre-built models with **working ExecuTorch export + AI Hub compile/quantize/profile recipes**. Use it two ways:
- **Reference recipe:** copy its export→profile flow into `model/aihub_profile.py` instead of building from scratch (`qai-hub-models-cli` exposes export/profile entry points).
- **Fallback / baseline:** if our custom model stalls on-site, profiling a stock audio model on the S25 still yields the NPU/latency screenshots for criterion #1 while we finish ours.

### 8.3 Confirmed AI Hub target devices (verified via `qai-hub list-devices`)

The hosted farm **includes the exact demo hardware**. Profile against the S25 Ultra so evidence matches the on-site loaner:

| Device (hosted) | OS | Chipset | Profile-job CLI |
|---|---|---|---|
| **Samsung Galaxy S25 Ultra** ⭐ primary | Android 15 | Snapdragon 8 Elite for Galaxy (`sm8750-ac`) | `--device "Samsung Galaxy S25 Ultra" --device-os 15` |
| Samsung Galaxy S25 | Android 15 | `sm8750-ac` | `--device "Samsung Galaxy S25" --device-os 15` |
| Snapdragon 8 Elite QRD (reference dev kit) | Android 15 | `qualcomm-snapdragon-8-elite` (`sm8750`) | `--device "Snapdragon 8 Elite QRD" --device-os 15` |

- **Canonical profile target = `Samsung Galaxy S25 Ultra`, OS 15** — same Snapdragon 8 Elite for Galaxy silicon as the loaner, so captured NPU/latency/memory numbers are demo-faithful.
- Newer **S26 / Snapdragon 8 Elite Gen 5** devices are also in the farm — ignore for v1; the demo device is the S25 Ultra.
- Every `submit_compile_job` / `submit_profile_job` / `submit_inference_job` **must pass this device** so the actual target's **Hexagon NPU** is exercised — this is the guard against the silent CPU-fallback trap in §11.

---

## 9. Testing and acceptance criteria

Use a lightweight **eval harness**, not heavy CI (right-sized for a 2-day build).

- **Model eval:** held-out accuracy/F1 on calm-vs-stressed; confusion matrix in `eval.py`. Acceptance: meaningfully better than chance and stable across a few voices.
- **Parity test:** PyTorch vs ExecuTorch vs on-device outputs agree within tolerance on the same input. Acceptance: max abs diff under a small epsilon.
- **NPU acceptance (criterion #1):** profile job shows inference on **NPU** (not CPU fallback); record per-window latency and peak memory. *Given X (audio window), when inference runs, then compute unit = NPU and latency < target.*
- **Privacy acceptance (criterion #3):** in airplane mode, app fully functions; network monitor shows **zero outbound requests**; no audio file written to storage.
- **Demo acceptance:** *Given a judge speaking a stressed sentence, when the meter crosses threshold, then the calming flow auto-launches within ~1s.*
- **Install acceptance (criterion #4):** teammate on a clean setup follows README and runs the app first try.

---

## 10. Build, run, and deployment instructions

Document these exact commands in the README:
- Model: `python model/export_executorch.py` → produces `assets/stress_model.pte`; `python model/aihub_profile.py` → prints profile/inference job URLs (paste screenshots into `benchmarks/`).
- App: open `android/` in Android Studio → Gradle sync → Build APK → install to S25 Ultra → launch. Provide an `adb install` one-liner too.
- Deployment target = the loaner S25 Ultra (devices are collected before judging, so **capture all evidence/video early** — don't leave proof on a device you'll hand back).

---

## 11. Edge cases, errors, and constraints

- **NPU fallback silently to CPU** — the classic trap. Verify compute unit explicitly via profiler; if it falls back, that quietly costs the 40%. Detect and call it out.
- **On-device vs offline feature mismatch** — log-mel params (n_fft, hop, mels, normalization) must match training exactly or accuracy collapses. Lock these constants in one shared spec.
- **Silence / no speech** — VAD must gate; don't classify silence as stress.
- **Noisy venue** (a loud hackathon room) — test in realistic noise; consider a simple noise-gate and a confidence floor. Have a quiet-ish demo spot.
- **One speaker vs many** — v1 assumes the user is the dominant voice; note as a limitation.
- **Threshold too jumpy** — EMA smoothing + hysteresis (different enter/exit thresholds) so the meter doesn't flicker.
- **Re-trigger spam** — lock the intervention while active.
- **Loaner device unknowns** — don't assume root or raw sensor access; build only on standard Android APIs.

---

## 12. Coding conventions and standards

- Kotlin official style; Python PEP 8 + type hints; format with ktlint / black.
- One shared constants file for audio/feature params (single source of truth).
- Small, reviewable commits; meaningful messages (helps the Copilot-Powered Build Award if you lean on Copilot — use it visibly and say so).
- Comment the *why*, especially around the ExecuTorch/QNN export and feature parity (judges read the repo for criterion #5).
- No secrets in git; `.gitignore` the token/config.

---

## 13. Dependencies, risks, and open questions

**Datasets (acute stress speech is scarce — map arousal):**
- RAVDESS, CREMA-D, TESS, SAVEE (emotional speech). Map high-arousal (angry/fearful) → "stressed," calm/neutral → "calm" as a pragmatic v1 label scheme.
- Study the provided **ShieldHer voice-stress model** as a reference/starting point.

**Key risks → mitigation:**
- *NPU export is the hardest, highest-value step* → start it during pre-work; grab a Qualcomm/Meta mentor on-site **first thing Day 1**; keep the XNNPACK CPU path as a working fallback.
- *2-day crunch* → arrive with tasks 1–3 essentially done (see §14). On-site = integration + optimization + polish, not training.
- *Accuracy unconvincing live* → curate a couple of reliable demo phrases that move the meter; show eval numbers on slides to back it up; be honest it's arousal-likelihood, not diagnosis.
- *Energy "evidence" is fuzzy* → argue it structurally (NPU ≫ CPU efficiency, INT8, VAD duty-cycling) and measure relative drain with Snapdragon Profiler / Android battery stats.

**Open questions to resolve (flag, don't guess):**
1. Exact ExecuTorch + QNN backend setup on the S25 Ultra — confirm with on-site mentors Day 1.
2. Final stress threshold + smoothing constants — tune on the actual device.
3. Do we keep a local session history at all, or store nothing? (Storing nothing is the stronger privacy story.)
4. Calming intervention scope — breathing + audio + haptics is enough; only add a local TTS/LLM coach if time remains.

---

## 14. Deliverables and milestones (time-phased — today is ~2 days out)

**Phase 0 — BEFORE June 27 (do this now; it's the difference between winning and scrambling):**
- `[CRITICAL]` Trained calm-vs-stressed PyTorch model + offline eval numbers.
- `[CRITICAL]` ExecuTorch `.pte` export working (XNNPACK), parity-checked.
- ✅ **DONE** — AI Hub Workbench account live & authenticated; toolchain installed and `qai-hub list-devices` verified (S25 Ultra target confirmed, §8.1/§8.3). **Remaining:** a dry-run compile/quantize/**profile** on the hosted S25 Ultra with **NPU confirmed** and latency/memory captured.
- `[IMPORTANT]` Android app shell: mic → ring buffer → VAD → log-mel, with a placeholder classifier.
- `[IMPORTANT]` Repo scaffold, README skeleton, LICENSE, demo-script outline.

**Day 1 on-site:**
- Sign loaner agreement; grab a mentor for ExecuTorch/QNN on real hardware.
- Land the model on device running inference on the **NPU**; wire UI meter; first end-to-end live loop.
- Build the calming intervention. Lock feature-param parity.

**Day 2 on-site:**
- Optimize: quantize/latency tune, hysteresis, noise handling.
- `[CRITICAL]` Airplane-mode + zero-network verification (record it on video).
- `[CRITICAL]` Capture all benchmark evidence and a backup demo video **before devices are collected**.
- Finalize README + repo, slides, and **rehearse the 5-minute demo** until it's boring-reliable.

**Final artifacts:** public GitHub repo (README with team names/emails, setup, run, OSS license) · cover image · slide deck · demo video · the running APK · `benchmarks/results.md` with NPU/latency/memory proof.

---

## The "extra layer": research, experimentation, and validation

**Deep research / prior-art (have answers ready for judges' questions):**
- *Why on-device:* voice is among the most sensitive biosignals — the privacy argument for local-only is airtight and is itself the use case.
- *Prior art:* cloud emotion APIs (privacy cost), passive wearables (detect but don't intervene). Our edge = **private + closed-loop (detect AND calm)** on commodity phone hardware.
- *Feasibility:* small log-mel CNNs quantize cleanly to INT8 and run in well under real-time on Hexagon — validated via AI Hub profiling.

**Experimentation — right-sized (don't over-engineer A/B for a hackathon):**
- Full A/B testing needs live traffic we won't have → instead run an **offline eval harness**: baseline vs quantized vs NPU model on a fixed test set, comparing accuracy, latency, and memory. That's the experiment judges respect at this stage.
- One worthwhile on-device "A/B": **CPU (XNNPACK) vs NPU (QNN)** latency and battery drain side-by-side — this single comparison is a *direct, visual proof* of the 40% technical thesis. Put it on a slide.

**Metrics & observability:** per-window latency, NPU utilization %, peak memory, model accuracy/F1, and relative power draw. Log latency in-app (debug overlay) so it's visible live.

**Validation/quality:** eval dataset + parity tests + regression check that quantization didn't tank accuracy.

**Security/privacy/compliance:** 100% on-device; raw audio discarded each window; no network permission used at runtime; OSS license; explicit mic-consent and a "not a medical device" disclaimer in-app.

**Performance/scalability:** target sub-300ms/window on NPU; VAD keeps duty cycle low for battery.

**Rollout/iteration (future-work slide):** per-user baseline calibration, multimodal fusion (camera-PPG HR, the ultrasonic-fingerprint pulse idea, keystroke dynamics), longitudinal trends — all on-device.

**Risk register:** see §13 (NPU fallback, crunch, live accuracy, energy evidence) each with an owner and mitigation.

---

### One-line pitch to open the demo with
> "This is Slashh. It hears stress in your voice and calms you down in real time — and watch this: it's in airplane mode. Nothing you say ever leaves the phone, because all of it runs on the Snapdragon NPU."