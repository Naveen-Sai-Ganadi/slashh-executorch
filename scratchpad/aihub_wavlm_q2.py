"""WavLM AI Hub INT8-on-NPU, take 2: quantize the STATIC-shape ONNX
(teacher_wavlm_broad_static.onnx, input fixed to [1,48000]) -> compile quantized
-> profile on Snapdragon NPU. Auth via ~/.qai_hub/client.ini (never printed)."""
import time
from pathlib import Path
import torch
import qai_hub as hub

STATIC_ONNX = Path("assets/teacher_wavlm_broad_static.onnx")
DEVICE = "Samsung Galaxy S24"
DEVICE_OS = "14"
CALIB = Path("assets/teacher_wavlm_broad_calib.pt")
N_CALIB = 16


def _url(j):
    return f"https://app.aihub.qualcomm.com/jobs/{j.job_id}"


def main():
    t0 = time.time()
    target = hub.Device(DEVICE, os=DEVICE_OS)
    blob = torch.load(CALIB, map_location="cpu", weights_only=False)
    calib = blob["calib"][:N_CALIB]

    print(f"[1/3] quantize job (INT8, {calib.shape[0]} calib) on static ONNX …", flush=True)
    qj = hub.submit_quantize_job(
        model=str(STATIC_ONNX),
        calibration_data={"wave": [calib[i:i+1].numpy() for i in range(calib.shape[0])]},
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    print(f"   quantize job: {qj.job_id}  {_url(qj)}", flush=True)
    qmodel = qj.get_target_model()
    print(f"   quantize DONE ({time.time()-t0:.0f}s)", flush=True)

    print("[2/3] compile quantized -> QNN context binary …", flush=True)
    cj = hub.submit_compile_job(
        model=qmodel, device=target,
        options="--target_runtime qnn_context_binary",
    )
    print(f"   qcompile job: {cj.job_id}  {_url(cj)}", flush=True)
    qbin = cj.get_target_model()
    print(f"   qcompile DONE ({time.time()-t0:.0f}s)", flush=True)

    print("[3/3] profile job …", flush=True)
    pj = hub.submit_profile_job(model=qbin, device=target)
    print(f"   profile job: {pj.job_id}  {_url(pj)}", flush=True)
    print("WAVLM_NPU_QPROFILE2_SUBMITTED", flush=True)
    print(f"   (quantize={qj.job_id} qcompile={cj.job_id} profile={pj.job_id})", flush=True)


if __name__ == "__main__":
    main()
