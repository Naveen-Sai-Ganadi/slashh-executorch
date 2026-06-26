"""Qualcomm AI Hub: compile → quantize(INT8) → profile StressNet on the NPU.

This is the script that produces the **40% evidence** (plan §3, §8.3): real
latency/throughput numbers from the Hexagon NPU on a Snapdragon 8 Elite, plus a
QNN ``.bin``/``.tflite`` context binary for on-device deployment.

⚠️  COST / CREDENTIAL SAFETY
    Submitting jobs uses your live AI Hub token (``~/.qai_hub/client.ini`` or
    ``QAI_HUB_API_TOKEN``) and consumes cloud credits. This script DRY-RUNS by
    default: it prints exactly what it would submit and exits. Pass ``--submit``
    to actually dispatch jobs. Never hard-code the token here.

Default target: Samsung Galaxy S25 Ultra (Snapdragon 8 Elite for Galaxy,
sm8750-ac, Android 15) — the canonical device from plan §8.3.

Pipeline once submitted:
    compile_job  : torch model + input spec  -> QNN context binary (.bin)
    quantize_job : calibration data          -> INT8 weights/activations
    profile_job  : compiled model on device  -> latency, throughput, NPU util
    inference_job: on-device output           -> parity vs local eager (optional)
"""

from __future__ import annotations

import argparse

import torch

from .audio_config import MODEL_INPUT_SHAPE
from .model import build_model

DEFAULT_DEVICE = "Samsung Galaxy S25 Ultra"
DEFAULT_DEVICE_OS = "15"


def _input_spec() -> dict:
    # qai_hub input-spec dict: name -> (shape, dtype)
    return {"logmel": (tuple(MODEL_INPUT_SHAPE), "float32")}


def _calibration_batch(n: int) -> torch.Tensor:
    """Small calibration set for INT8 — real run should use held-out features."""
    from .data import synthetic_dataset

    x, _ = synthetic_dataset(max(1, n // 2), seed=4242)
    return x[:n]


def describe(weights: str | None, device: str, device_os: str, quantize: bool) -> None:
    spec = _input_spec()
    print("AI Hub job plan (DRY RUN — nothing submitted)")
    print(f"  device      : {device}  (--device-os {device_os})")
    print(f"  weights     : {weights or '<random init>'}")
    print(f"  input spec  : {spec}")
    print(f"  quantize    : {'INT8 (compile + quantize + profile)' if quantize else 'fp16 (compile + profile)'}")
    print("  steps       : trace -> submit_compile_job"
          + (" -> submit_quantize_job" if quantize else "")
          + " -> submit_profile_job")
    print("\nRe-run with --submit to dispatch (uses live token, consumes credits).")


def submit(weights: str | None, device: str, device_os: str, quantize: bool,
           calib_n: int) -> None:
    import qai_hub as hub

    model = build_model(weights)
    example = torch.randn(*MODEL_INPUT_SHAPE)
    traced = torch.jit.trace(model, example)

    target = hub.Device(device, os=device_os)

    print(f"submitting compile job → {device} (os {device_os}) …")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=target,
        input_specs={"logmel": tuple(MODEL_INPUT_SHAPE)},
    )
    compiled = compile_job.get_target_model()
    print(f"  compile job: {compile_job.job_id}")

    if quantize:
        calib = _calibration_batch(calib_n)
        print(f"submitting quantize job (INT8, {calib.shape[0]} calib samples) …")
        quantize_job = hub.submit_quantize_job(
            model=compiled,
            calibration_data={"logmel": [calib[i:i + 1].numpy() for i in range(calib.shape[0])]},
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
        )
        compiled = quantize_job.get_target_model()
        print(f"  quantize job: {quantize_job.job_id}")

    print("submitting profile job …")
    profile_job = hub.submit_profile_job(model=compiled, device=target)
    print(f"  profile job: {profile_job.job_id}")
    print("\nView results in the AI Hub Workbench. Record latency/throughput in")
    print("docs/benchmarks (plan §9) once the jobs finish.")


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Hub compile/quantize/profile StressNet")
    ap.add_argument("--weights", "-w", default="assets/stress_model.pt")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--device-os", default=DEFAULT_DEVICE_OS)
    ap.add_argument("--no-quantize", action="store_true",
                    help="profile fp16 instead of INT8")
    ap.add_argument("--calib-n", type=int, default=32,
                    help="INT8 calibration sample count")
    ap.add_argument("--submit", action="store_true",
                    help="ACTUALLY submit jobs (uses live token, consumes credits)")
    args = ap.parse_args()

    quantize = not args.no_quantize
    if args.submit:
        submit(args.weights, args.device, args.device_os, quantize, args.calib_n)
    else:
        describe(args.weights, args.device, args.device_os, quantize)


if __name__ == "__main__":
    main()
