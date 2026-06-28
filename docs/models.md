# Models

Four models, three on-device. This doc covers how each is trained/exported and how to retune the
one parameter most worth tuning (the fusion's conservativeness).

| Model | Role | Where it runs | Artifact |
|-------|------|---------------|----------|
| **Whisper-Tiny** | speech → text (ASR) | Hexagon NPU (QNN context binary, resident daemon) | `encoder.bin` / `decoder.bin` (AI Hub) |
| **WavLM** | audio stress (prosody) | Hexagon NPU (QNN context binary, `qnn-net-run`) | `forward_2.bin` |
| **Text classifier** | word stress (semantics) | on-device CPU (ExecuTorch XNNPACK) | `assets/text_stress.pte` + `text_stress_vocab.txt` |
| **Fusion** | combine audio + text | on-device CPU (ExecuTorch XNNPACK) | `assets/fusion.pte` |

> The Python side uses the project venv (Python 3.11, `torch`, `executorch`, `scikit-learn`,
> `pandas`). The shipped `.pte`s are committed — you only need this to retrain/re-export.

## Text stress classifier (`model/text_stress.py`)

Trained on the **[Dreaddit](https://aclanthology.org/D19-6213/)** corpus — ~3.5 k human-labeled
Reddit posts (stress vs not-stress). The features are deliberately tiny so the whole thing is a
small on-device `.pte` with **no LLM**:

- **Tokenize** (`model/text_features.py`, the single source of truth mirrored by `TextFeatures.kt`):
  lowercase → replace every non-`a-z` run with a space → unigrams.
- **Vectorize**: counts over a fixed **2000-word vocabulary** (top tokens by document frequency in
  the training split), **L2-normalized** → `[1, 2000]` float32.
- **Model**: `Linear(2000→64) → ReLU → Linear(64→1) → sigmoid`. Test accuracy ≈ **70.9%** (well
  above the ~52% majority baseline).

```bash
python -m model.text_stress     # → assets/text_stress.pte + android/.../text_stress_vocab.txt + checkpoint
```

The Kotlin and Python tokenizers must stay byte-for-byte identical — a JVM parity test
(`TextFeaturesParityTest`) asserts this against golden vectors in
`android/app/src/test/resources/golden_text_fusion.json`.

It learns *cues*, not keywords: *"overwhelmed", "anxious", "scared", "deadline"* push the score up;
*"relaxed", "calm", "garden"* push it down. (Aggression/profanity isn't "stress" in Dreaddit's
sense, so it scores those low — by design.)

## Fusion perceptron (`model/fusion.py`)

A **single linear perceptron** `Linear(2,1) → sigmoid` over `[audio_score, text_score]`.

Two deliberate choices make it behave intuitively:

1. **Symmetric** — training pairs are **swap-augmented** (each `(audio, text)` pair is also added as
   `(text, audio)` with the same label), so the two inputs weigh **equally**. Neither modality
   dominates: `fuse(1.0, 0.25) == fuse(0.25, 1.0)`.
2. **Conservative boundary** — after training, the bias is set so the output crosses 0.5 only when
   **`audio + text ≥ DECISION_BOUNDARY_SUM`** (default **1.2** → both ≈ 0.6+). This is the single
   knob for how trigger-happy it is.

Training pairs come from a late-fusion-under-conditional-independence model: text-score
distributions from running the text model over all Dreaddit rows, and audio-score distributions
(Beta(2.4,5.6)/Beta(5.6,2.4) — the documented fallback, because the v1 StressNet checkpoint is
degenerate on RAVDESS). Sample grid of the shipped model:

```
fuse(0.5,0.5)=0.14   fuse(0.6,0.6)=0.50   fuse(0.7,0.7)=0.86   fuse(0.8,0.8)=0.98
```

```bash
python -m model.fusion              # → assets/fusion.pte + checkpoint + prints the 5×5 grid
python -m model.golden_text_fusion  # regenerate golden vectors after any change
cp model/golden_text_fusion.json android/app/src/test/resources/golden_text_fusion.json
```

### Retuning trigger-happiness (the one knob)

Seeing too many false "stressed" calls? Raise it. Not firing when you're clearly stressed? Lower it.

```python
# model/fusion.py
DECISION_BOUNDARY_SUM = 1.2   # ↑ more conservative (e.g. 1.4) · ↓ more sensitive (e.g. 1.0)
```

Then `python -m model.fusion` to re-export. Pair it with the latch in `StressPipeline.kt`
(`enterThreshold` / `releaseThreshold`) and the in-app **Sensitivity** slider.

## Whisper-Tiny (ASR)

Used **as-is** from Qualcomm AI Hub's Whisper-Tiny QNN context-binary export for the Snapdragon
8 Elite — we don't retrain it. The on-device runner (`tools/whisper/whisper_qnn.cpp`, committed)
implements the mel front-end, the autoregressive greedy decode loop, and the byte-level-BPE
detokenizer in pure C++ around the QNN C-API, so the full **PCM → text** path runs on-device. See
[NPU runtime](npu-runtime.md) for obtaining `encoder.bin` / `decoder.bin`.

## WavLM (audio) & the v1 StressNet

- **WavLM** is the audio-stress (arousal) model, lowered to a QNN context binary and run via
  `qnn-net-run`. See the WavLM-on-NPU proof under [`v2/`](v2/).
- **StressNet** is the original v1 tiny log-mel CNN (`model/model.py`), trained on RAVDESS and
  exported with `model/export_executorch.py` (`assets/stress_model.pte`). It is **not** in the live
  loop on a retail S25 (its in-app forward SIGSEGVs); it remains the host-parity reference and the
  AI-Hub NPU-utilization evidence. The original StressNet train/eval/export docs are in
  [`plan.md`](plan.md) and [`benchmarks/`](benchmarks/).
