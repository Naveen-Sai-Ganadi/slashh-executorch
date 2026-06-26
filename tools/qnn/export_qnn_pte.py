"""Export StressNet to a QNN/Hexagon-delegated ExecuTorch .pte for the S25.

Runs ONLY on Linux x86_64 (the QNN backend ships a compiled native module and
ExecuTorch auto-downloads the QNN SDK there). Use tools/qnn/build_qnn_pte.sh to
run this inside Docker from macOS.

Target: Samsung Galaxy S25 / Snapdragon 8 Elite = QcomChipset.SM8750 (HTP V79).
Produces a w8a8-quantized program (PT2E + QnnQuantizer, calibrated on RAVDESS);
falls back to FP16-on-HTP if the quantizer flow is unavailable. Either runs on
the NPU; quantized is smaller/faster.

Output: android/app/src/main/assets/stress_model_qnn.pte
"""

import sys
import torch

from model.audio_config import MODEL_INPUT_SHAPE
from model.model import build_model

OUT = "android/app/src/main/assets/stress_model_qnn.pte"
WEIGHTS = "assets/stress_model.pt"


def _calibration(n=32):
    """Real RAVDESS features if present, else random (ranges only matter for PTQ)."""
    try:
        from model.data import ravdess_dataset
        x, _, _ = ravdess_dataset("data")
        print(f"[calib] using {min(n, x.shape[0])} real RAVDESS samples")
        return x[:n]
    except Exception as e:  # noqa: BLE001
        print(f"[calib] RAVDESS unavailable ({e}); using random calibration")
        return torch.randn(n, *MODEL_INPUT_SHAPE[1:])


def main():
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )

    model = build_model(WEIGHTS).eval()
    example = (torch.randn(*MODEL_INPUT_SHAPE),)

    quantized_model = model
    use_fp16 = True
    try:
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

        print("[quant] PT2E + QnnQuantizer (w8a8) ...")
        captured = torch.export.export(model, example).module()
        quantizer = QnnQuantizer()
        prepared = prepare_pt2e(captured, quantizer)
        calib = _calibration()
        with torch.no_grad():
            for i in range(0, calib.shape[0]):
                prepared(calib[i:i + 1])
        quantized_model = convert_pt2e(prepared)
        use_fp16 = False
        print("[quant] quantization OK -> w8a8 on HTP")
    except Exception as e:  # noqa: BLE001
        print(f"[quant] quantizer flow unavailable ({e}); exporting FP16 on HTP")

    backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8750,            # Snapdragon 8 Elite (S25), HTP V79
        backend_options=backend_options,
    )
    edge = to_edge_transform_and_lower_to_qnn(quantized_model, example, compiler_specs)
    prog = edge.to_executorch()

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(prog.buffer)
    print(f"[done] wrote {OUT}  ({len(prog.buffer):,} bytes, "
          f"{'w8a8' if not use_fp16 else 'fp16'}, SM8750/V79)")

    _extract_qnn_runtime_libs()


def _extract_qnn_runtime_libs():
    """Copy the on-device QNN runtime .so's (arm64 + Hexagon V79 skel) out of the
    auto-downloaded SDK into the app's jniLibs, so they ship in the APK."""
    import glob
    import os
    import shutil

    roots = []
    if os.getenv("QNN_SDK_ROOT"):
        roots.append(os.environ["QNN_SDK_ROOT"])
    import executorch.backends.qualcomm as q
    roots += [os.path.join(p, "sdk", "qnn") for p in q.__path__]
    roots += glob.glob(os.path.expanduser("~/.cache/executorch/qnn*"))
    roots += glob.glob(os.path.expanduser("~/.cache/executorch/qnn/**/"), recursive=True)

    dest = "android/app/src/main/jniLibs/arm64-v8a"
    os.makedirs(dest, exist_ok=True)
    patterns = [
        "lib/aarch64-android/libQnn*.so",
        "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so",
        "lib/hexagon-v79/unsigned/libQnnHtpV79.so",
    ]
    copied = 0
    seen = set()
    for root in roots:
        for pat in patterns:
            for src in glob.glob(os.path.join(root, "**", pat), recursive=True):
                base = os.path.basename(src)
                if base in seen:
                    continue
                shutil.copy2(src, os.path.join(dest, base))
                seen.add(base)
                copied += 1
                print(f"[libs] {base}")
    if copied:
        print(f"[libs] copied {copied} QNN runtime libs -> {dest}")
    else:
        print("[libs] WARNING: no QNN runtime libs found; check SDK layout. "
              "Roots tried:", roots[:3])


if __name__ == "__main__":
    sys.exit(main())
