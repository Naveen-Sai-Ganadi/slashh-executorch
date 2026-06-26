# Slashh AI — on-device voice-stress detection

Detect stress in your voice **entirely on-device**, in real time, with no network.
A small CNN runs on the phone's audio, scores each 3-second window for vocal
stress, and offers a calming intervention when stress sustains — all in
airplane mode. The model is trained in PyTorch, exported to an ExecuTorch
`.pte`, and lowered to the Snapdragon Hexagon NPU via Qualcomm AI Hub.

- **Private by construction** — audio never leaves the device; the inference
  hot path has no network dependency.
- **Fast + efficient** — INT8 on the NPU (the headline evidence) with an
  XNNPACK-CPU path as the portable safety net.
- **Reproducible** — train → export → run works on a clean checkout in seconds.

See [`docs/plan.md`](docs/plan.md) for the full design and
[`docs/autonomy/BACKLOG.md`](docs/autonomy/BACKLOG.md) for the roadmap.

## Architecture

```
mic 16kHz ─▶ VAD ─▶ log-mel  ─▶  [ ExecuTorch .pte ]  ─▶ score ─▶ EMA + hysteresis ─▶ meter / calm cue
            (3s window,        StressNet: 3×conv→GAP      [0,1]
             1s hop)            →linear→sigmoid
```

The `.pte` is **classifier-only**: it takes a fixed `[1, 1, 64, 301]` log-mel
tensor and returns one stress probability. Feature extraction (log-mel) runs
natively outside the graph, so only delegate-friendly ops (conv/bn/relu/pool/
linear/sigmoid) lower to the NPU. `model/audio_config.py` is the single source
of truth for every audio constant, shared by training and the on-device
extractor so features agree at train and inference time.

## Layout

| Path | What |
|------|------|
| `model/audio_config.py` | sample rate, window/hop, mel params, thresholds — one source of truth |
| `model/features.py` | golden log-mel extractor (the on-device one must match) |
| `model/model.py` | `StressNet` definition + `build_model` / `example_input` |
| `model/data.py` | synthetic dataset (offline) + real `calm/` `stressed/` folder loader |
| `model/train.py` | train → checkpoint (`{"model": state_dict, "meta": …}`) |
| `model/eval.py` | accuracy/precision/recall + optional `.pte` parity |
| `model/export_executorch.py` | `nn.Module` → XNNPACK `.pte` |
| `model/run_pte.py` | run a `.pte` through the ExecuTorch runtime (host) |
| `model/aihub_profile.py` | AI Hub compile/quantize(INT8)/profile on the NPU |
| `model/golden.py` | emit golden log-mel vectors for the Android parity test |
| `tests/` | parity (`.pte` == eager, 1e-4) + feature contract |
| `android/` | Kotlin app: mic → VAD → log-mel → `.pte` → meter (see [`android/README.md`](android/README.md)) |

## Setup

Python 3.11. Pins in `model/requirements.txt` match the verified toolchain.

```bash
uv venv --python 3.11
uv pip install -r model/requirements.txt
```

## Quickstart

```bash
# 1. train a model (synthetic, offline, ~seconds) → checkpoint + .pte
python -m model.train --epochs 8 --out assets/stress_model.pt \
    --export assets/stress_model.pte

# 2. run the exported model through the ExecuTorch runtime
python -m model.run_pte --model assets/stress_model.pte

# 3. evaluate the checkpoint and fleet-parity-check the .pte
python -m model.eval --weights assets/stress_model.pt --pte assets/stress_model.pte

# 4. tests: exported .pte matches eager within 1e-4 + feature contract
python -m pytest tests/ -q
```

Steps 1–2 are exercised end-to-end on every run by `tests/test_quickstart.py`,
so a change that breaks this documented path fails CI rather than a fresh
checkout.

Train on real data (RAVDESS/CREMA-D/TESS/SAVEE mapped to arousal, per plan §13)
by arranging `wav`s as `<root>/calm/*.wav` and `<root>/stressed/*.wav`:

```bash
python -m model.train --data-dir data/arousal --epochs 40
```

## Android app

The on-device pipeline lives in [`android/`](android/README.md): a Kotlin
implementation of mic capture → VAD → log-mel → ExecuTorch inference → EMA +
hysteresis meter. The manifest declares **no `INTERNET` permission**, so the
whole loop runs in airplane mode. `android/app/src/main/java/ai/slashh/audio/`
reimplements `model/features.py` natively (exact 400-pt DFT + HTK mel
filterbank); a JVM parity test asserts it matches torchaudio within 1e-3 on the
golden vectors emitted by `python -m model.golden`.

```bash
cd android
gradle wrapper                 # first time only
./gradlew testDebugUnitTest    # log-mel parity + VAD + pipeline (no device)
./gradlew installDebug         # build + install to a connected device
```

## Snapdragon NPU profiling (Qualcomm AI Hub)

`model/aihub_profile.py` compiles, INT8-quantizes, and profiles the model on a
**Samsung Galaxy S25 Ultra** (Snapdragon 8 Elite, sm8750-ac, Android 15). It
**dry-runs by default** — submitting jobs uses your live AI Hub token and
consumes credits:

```bash
python -m model.aihub_profile                 # dry run — prints the plan
python -m model.aihub_profile --submit        # actually dispatch (uses credits)
```

Credentials live in `~/.qai_hub/client.ini` or `QAI_HUB_API_TOKEN` (outside the
repo). **Never commit the token.**

## License

See [LICENSE](LICENSE).
