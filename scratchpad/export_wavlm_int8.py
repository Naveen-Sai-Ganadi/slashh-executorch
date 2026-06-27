"""INT8 (PT2E + XNNPACK) .pte for the merged WavLM teacher, so it fits the
emulator's RAM (~317MB vs the 1.2GB fp32 build that OOMs a 3GB emulator).

Static post-training quant, host-only (NOT the AI Hub/NPU path):
    export -> prepare_pt2e -> calibrate(raw waves) -> convert_pt2e
            -> export -> XNNPACK lower -> .pte

Calibrated on RAW 16 kHz waves (DeployTeacher normalizes in-graph), matching how
the model is fed on device. Then run through the ExecuTorch runtime on host and
report INT8-vs-eager parity + size.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/export_wavlm_int8.py
"""
import time
from pathlib import Path

import torch

from model.export_teacher import _load_and_merge
from model.export_executorch import _ensure_flatc
from model.train_lora import load_waves, _split_idx

ASSETS = Path("assets")
N_CALIB = 24


def main():
    t0 = time.time()
    _ensure_flatc()

    eager, deploy, tag = _load_and_merge("wavlm", "broad")
    deploy = deploy.eval()
    print(f"[{tag}] merged ({time.time()-t0:.0f}s)", flush=True)

    waves = load_waves()
    tr, va, _y = _split_idx("broad")
    g = torch.Generator().manual_seed(2024)
    calib = waves[tr[torch.randperm(len(tr), generator=g)[:N_CALIB]]].contiguous()
    ex = (calib[:1].clone().contiguous(),)  # real wave, not zeros (avoids std=0 / inf)

    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    captured = torch.export.export(deploy, ex).module()
    print(f"[{tag}] captured ({time.time()-t0:.0f}s)", flush=True)
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for i in range(calib.shape[0]):
            prepared(calib[i:i + 1])
    print(f"[{tag}] calibrated {N_CALIB} ({time.time()-t0:.0f}s)", flush=True)
    converted = convert_pt2e(prepared)

    exported = torch.export.export(converted, ex)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    buf = edge.to_executorch().buffer
    out = ASSETS / f"teacher_{tag}_int8.pte"
    out.write_bytes(buf)
    print(f"[{tag}] INT8 .pte -> {out} ({len(buf)/1e6:.0f} MB, {time.time()-t0:.0f}s)",
          flush=True)

    # parity: eager fp32 vs INT8 .pte through the ExecuTorch runtime
    from model.run_pte import run_pte
    real = waves[va[:8]].contiguous()
    agree, dmax = 0, 0.0
    with torch.no_grad():
        for i in range(real.shape[0]):
            r = deploy(real[i:i + 1]).flatten()
            q = run_pte(str(out), real[i:i + 1]).flatten()
            dmax = max(dmax, (r - q).abs().max().item())
            agree += int((r > 0).item() == (q > 0).item())
    print(f"[{tag}] INT8 vs fp32: max|Δ|={dmax:.3f} decision-agree={agree}/8 "
          f"({time.time()-t0:.0f}s)", flush=True)
    print("OK_WAVLM_INT8", flush=True)


if __name__ == "__main__":
    main()
