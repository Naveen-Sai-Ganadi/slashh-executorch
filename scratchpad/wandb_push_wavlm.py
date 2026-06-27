"""Log the NPU-ready WavLM model to a Weights & Biases Artifact for sharing.

Auth: run `wandb login` yourself first (writes ~/.netrc; this script never sees
the key). Then:

    PYTHONPATH="$PWD" .venv/bin/python3 scratchpad/wandb_push_wavlm.py \
        --entity <your-wandb-entity> --project slashh-stress

Logs the deployable QNN context binary (+ the bias-patched ONNX and the Hexagon
profile JSON, unless --bin-only) as a versioned artifact named `wavlm-npu-int8`.
A colleague with access to the project pulls it with:

    import wandb
    art = wandb.Api().artifact("<entity>/slashh-stress/wavlm-npu-int8:latest")
    art.download()
"""
import argparse
from pathlib import Path

import wandb

ASSETS = Path("assets")
BIN = ASSETS / "teacher_wavlm_broad_npu_int8_qnn.bin"   # QNN context binary (Hexagon)
ONNX = ASSETS / "teacher_wavlm_broad_static_bias.onnx"  # bias-patched source
PROFILE = Path("scratchpad/jglo88klg_profile/profile.json")

# Provenance recorded on the artifact (from this session's AI Hub run).
META = {
    "device": "Samsung Galaxy S24 (Snapdragon, Hexagon NPU)",
    "device_os": "14",
    "runtime": "qnn_context_binary",
    "precision": "INT8 (weights + activations)",
    "input_shape": [1, 48000],
    "npu_offload_pct": 100.0,
    "ops_on_npu": "1302/1302",
    "est_inference_ms": 7298,
    "peak_inference_mem_mb": 105,
    "warm_load_ms": 279,
    "quantize_job": "jgko06925",
    "compile_job": "j5qz14yn5",
    "profile_job": "jglo88klg",
    "fix": "zero-bias injection on 7 bias-less feature-extractor Conv1d (numerically inert)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None, help="W&B entity; default = the key's default entity")
    ap.add_argument("--project", default="slashh-stress")
    ap.add_argument("--name", default="wavlm-npu-int8")
    ap.add_argument("--with-onnx", action="store_true", help="also upload the 1.26GB bias-patched ONNX source")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"missing {BIN}")

    run = wandb.init(entity=args.entity, project=args.project,
                     job_type="publish-model", config=META)
    art = wandb.Artifact(
        args.name, type="model",
        description="WavLM LoRA-merged, INT8 QNN context binary — 100% Hexagon NPU offload.",
        metadata=META,
    )
    art.add_file(str(BIN), name="wavlm_npu_int8_qnn.bin")
    print(f"+ {BIN.name} ({BIN.stat().st_size/1e6:.0f} MB)", flush=True)
    if PROFILE.exists():
        art.add_file(str(PROFILE), name="hexagon_profile.json")
        print(f"+ {PROFILE.name}", flush=True)
    if args.with_onnx and ONNX.exists():
        art.add_file(str(ONNX), name="wavlm_static_bias.onnx")
        print(f"+ {ONNX.name} ({ONNX.stat().st_size/1e9:.2f} GB)", flush=True)

    run.log_artifact(art)
    art.wait()
    print(f"\nlogged artifact: {args.entity}/{args.project}/{args.name}:{art.version}", flush=True)
    print(f"colleague pulls: wandb.Api().artifact('{args.entity}/{args.project}/{args.name}:latest').download()", flush=True)
    run.finish()
    print("WANDB_PUSH_DONE", flush=True)


if __name__ == "__main__":
    main()
