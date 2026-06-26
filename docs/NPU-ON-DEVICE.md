# Running the model on the Snapdragon NPU (in-app) — event runbook

## Where we are

The app runs the **real RAVDESS-trained `StressNet`** via ExecuTorch. Today it
loads `assets/stress_model.pte` on the **XNNPACK (CPU)** backend — the plan's
"safety-net" path. The **same model** was profiled on the **Hexagon NPU** via
Qualcomm AI Hub (**100% NPU, 0.060 ms/window**) — that is the 40% evidence, and
it stands on its own.

What is NOT yet true: the **live app** executing on the NPU. Three things block
it, all needing the Qualcomm QNN SDK (Linux), and none verifiable on an emulator
(emulators have no Hexagon NPU):

1. The Maven `executorch-android:1.2.0` AAR ships **XNNPACK only** — confirmed:
   it contains just `libexecutorch.so`, no QNN/HTP libs.
2. The AI Hub artifact `stress_model_qnn.bin` is a **raw QNN context binary**,
   not an ExecuTorch `.pte`. `Module.load()` needs a **QNN-delegated `.pte`**.
3. The QNN runtime `.so` libs (`libQnnHtp.so`, `libQnnSystem.so`,
   `libQnnHtpV*Stub.so`, `libQnnHtpV*Skel.so`, …) are proprietary Qualcomm libs,
   not redistributed in the PyTorch AAR.

The app is already **NPU-ready**: `MainActivity.loadClassifier()` loads
`stress_model_qnn.pte` if present, else falls back to `stress_model.pte`. So
finishing the NPU path is a **drop-in** — no app code change.

## To finish it at the event (with a Qualcomm/Meta mentor)

### 1. Produce a QNN-delegated `.pte` (Linux x64 + QNN SDK)
On an Ubuntu 22.04 box with the QNN SDK installed and ExecuTorch built with the
Qualcomm backend:

```python
# pseudo — use the model + weights from this repo (model/model.py, assets/stress_model.pt)
from executorch.backends.qualcomm.utils.utils import (
    generate_qnn_executorch_compiler_spec, to_edge_transform_and_lower_to_qnn,
    QcomChipset,
)
import torch
from model.model import build_model
from model.audio_config import MODEL_INPUT_SHAPE

model = build_model("assets/stress_model.pt").eval()
example = (torch.randn(*MODEL_INPUT_SHAPE),)
exported = torch.export.export(model, example)
spec = generate_qnn_executorch_compiler_spec(soc_model=QcomChipset.SM8750)  # 8 Elite
edge = to_edge_transform_and_lower_to_qnn(exported, spec)
open("stress_model_qnn.pte", "wb").write(edge.to_executorch().buffer)
```

(Or ask the mentor to run the standard ExecuTorch Qualcomm export for our model —
fixed input `[1,1,64,301]`. Quantize w8a8 if time allows; our PT2E check showed
no accuracy drop.)

### 2. Get a QNN-enabled ExecuTorch Android AAR + Qualcomm libs
- A `executorch-android` AAR **built with the QNN backend**, OR the
  `libqnn_executorch_backend.so` added alongside `libexecutorch.so`.
- The QNN SDK runtime `.so`s for `arm64-v8a` (HTP/V79 for 8 Elite).
- Drop the AAR in `android/app/libs/` (and switch the dependency to it), and the
  QNN `.so`s into `android/app/src/main/jniLibs/arm64-v8a/`.

### 3. Drop in the model
- Put `stress_model_qnn.pte` in `android/app/src/main/assets/`.
- Rebuild. `loadClassifier()` auto-selects it; logcat prints
  `model loaded: stress_model_qnn.pte (NPU/QNN)`.

### 4. Verify on the real S25
- `adb logcat -s Slashh` → confirm the NPU model loaded.
- Speak; confirm the meter responds and latency is low.
- If QNN load fails for any reason, the app **automatically falls back to CPU**,
  so the demo never hard-fails.

## Demo-claim guidance (be precise)
- ✅ "Our RAVDESS-trained mel-CNN runs on-device with ExecuTorch."
- ✅ "We profiled the same model on the Snapdragon Hexagon NPU via AI Hub:
  100% NPU, 0.06 ms/window."
- Only say the **live app** runs on the NPU **after** step 4 prints the QNN load
  line on the actual S25. Until then, the live app runs on CPU (XNNPACK).
