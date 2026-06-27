"""Submit the merged WavLM teacher to Qualcomm AI Hub: compile -> quantize(INT8)
-> profile on a Snapdragon NPU. Answers "is WavLM NPU-compatible / how much
offloads to Hexagon?" — the teacher analogue of the M5 StressNet run.

Mirrors model/aihub_profile.submit but for the raw-waveform [1,48000] teacher.
Auth via ~/.qai_hub/client.ini (NEVER printed). Spends live credits — human
authorized (all-models-on-npu directive + explicit instruction).

    PYTHONPATH="$PWD" .venv/bin/python3 scratchpad/aihub_wavlm.py
"""
import time
from pathlib import Path

import torch
import qai_hub as hub

ASSETS = Path("assets")
ONNX = ASSETS / "teacher_wavlm_broad.onnx"          # transformer -> QNN: ONNX is the robust path
TS = ASSETS / "teacher_wavlm_broad.ts.pt"
CALIB = ASSETS / "teacher_wavlm_broad_calib.pt"
DEVICE = "Samsung Galaxy S24"                        # soc-model:57, matches M5 StressNet evidence
DEVICE_OS = "14"
SHAPE = (1, 48000)
N_CALIB = 16                                         # keep quantize calib modest (cost/time)


def _url(job):
    return f"https://app.aihub.qualcomm.com/jobs/{job.job_id}"


def main():
    t0 = time.time()
    target = hub.Device(DEVICE, os=DEVICE_OS)
    src = ONNX if ONNX.exists() else TS
    print(f"[wavlm-npu] source={src.name} ({src.stat().st_size/1e6:.0f} MB) "
          f"device={DEVICE} os={DEVICE_OS}", flush=True)

    # 1) COMPILE  (fp32 -> AI Hub target; QNN context binary for the NPU)
    print("[1/3] submitting compile job (QNN context binary) …", flush=True)
    compile_job = hub.submit_compile_job(
        model=str(src),
        device=target,
        input_specs={"wave": SHAPE},
        options="--target_runtime qnn_context_binary",
    )
    print(f"   compile job: {compile_job.job_id}  {_url(compile_job)}", flush=True)
    compiled = compile_job.get_target_model()   # blocks until compile done
    print(f"   compile DONE ({time.time()-t0:.0f}s)", flush=True)

    # 2) QUANTIZE  (INT8 weights + activations, calibrated on real waves)
    blob = torch.load(CALIB, map_location="cpu", weights_only=False)
    calib = blob["calib"][:N_CALIB]
    print(f"[2/3] submitting quantize job (INT8, {calib.shape[0]} calib) …", flush=True)
    quantize_job = hub.submit_quantize_job(
        model=compiled,
        calibration_data={"wave": [calib[i:i+1].numpy() for i in range(calib.shape[0])]},
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    print(f"   quantize job: {quantize_job.job_id}  {_url(quantize_job)}", flush=True)
    qmodel = quantize_job.get_target_model()
    print(f"   quantize DONE ({time.time()-t0:.0f}s)", flush=True)

    # 2b) re-compile the quantized model to a QNN context binary for the device
    print("[2b] compiling quantized model -> QNN context binary …", flush=True)
    qcompile = hub.submit_compile_job(
        model=qmodel,
        device=target,
        options="--target_runtime qnn_context_binary",
    )
    print(f"   qcompile job: {qcompile.job_id}  {_url(qcompile)}", flush=True)
    qbin = qcompile.get_target_model()
    print(f"   qcompile DONE ({time.time()-t0:.0f}s)", flush=True)

    # 3) PROFILE  (on-device latency + per-layer NPU/GPU/CPU offload breakdown)
    print("[3/3] submitting profile job …", flush=True)
    profile_job = hub.submit_profile_job(model=qbin, device=target)
    print(f"   profile job: {profile_job.job_id}  {_url(profile_job)}", flush=True)
    print("WAVLM_NPU_JOBS_SUBMITTED", flush=True)
    print(f"   (compile={compile_job.job_id} quantize={quantize_job.job_id} "
          f"qcompile={qcompile.job_id} profile={profile_job.job_id})", flush=True)


if __name__ == "__main__":
    main()
