"""Surgical INT8 for WavLM: quantize the whole net EXCEPT the relative-position
attention path that crashes PT2E.

Earlier finding: a global PT2E pass crashes during calibration with
    "Tensor dtype mismatch! Expected: Long, Got: float"
at scaled_dot_product_attention, because observers land on WavLM's gated
relative-position bias path (gru_rel_pos_linear -> bias -> SDPA mask). That
subgraph is tiny (one Linear + one Embedding per layer); everything else
(q/k/v/out_proj + the param-heavy feed-forward linears) is ordinary and
quantizes fine.

XNNPACKQuantizer.set_filter_function(fn) keeps only nodes where fn(node)==True.
So set_global(int8) + a filter that REJECTS nodes in the rel-pos modules =
"quantize all linears except the relative-position ones".

Fallback ladder (broadest scope that actually builds wins):
  A) exclude only the rel-pos path            (max coverage)
  B) exclude rel-pos + all attention linears  (FFN-only; safest)

Reports for the winning scope: INT8 .pte size, and INT8-vs-fp32 parity +
decision-agreement through the ExecuTorch runtime on host.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/export_wavlm_int8_surgical.py
"""
import time
from pathlib import Path

import torch

from model.export_teacher import _load_and_merge
from model.export_executorch import _ensure_flatc
from model.train_lora import load_waves, _split_idx

ASSETS = Path("assets")
N_CALIB = 16

# module-path substrings whose Linear nodes must stay fp32
RELPOS = ("gru_rel_pos_linear", "rel_attn_embed")
ATTN = ("attention.k_proj", "attention.q_proj", "attention.v_proj",
        "attention.out_proj")


def _node_paths(n):
    """All module paths in a node's nn_module_stack (lowest-level last)."""
    stack = n.meta.get("nn_module_stack")
    if not stack:
        return []
    return [v[0] if isinstance(v, (tuple, list)) else str(v) for v in stack.values()]


def make_filter(exclude_substrings):
    def keep(n):
        paths = _node_paths(n)
        return not any(s in p for p in paths for s in exclude_substrings)
    return keep


def build(deploy, calib, ex, exclude, tag_suffix, t0):
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    # WavLM's relative-position bucket math uses an int scalar as a Long index.
    # prepare_pt2e unconditionally runs transform_for_annotation ->
    # _convert_scalars_to_attrs, which rewrites that scalar to a FLOAT tensor
    # attr and crashes the index op. We only quantize Linears (whose correctness
    # doesn't need scalar->attr), so skip that transform.
    class SafeQuantizer(XNNPACKQuantizer):
        def transform_for_annotation(self, model):
            return model

    # PER-TENSOR (not per-channel): the q/k/v projections feed WavLM's
    # hand-rolled attention matmuls, which XNNPACK won't partition, so their
    # weight-dequant stays in the graph. dequantize_per_channel has no portable
    # out-variant (serialization fails); dequantize_per_tensor does -> it runs.
    captured = torch.export.export(deploy, ex).module()
    quantizer = SafeQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False))
    quantizer.set_filter_function(make_filter(exclude))
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for i in range(calib.shape[0]):
            prepared(calib[i:i + 1])           # <-- this is where global-scope crashed
    print(f"  [{tag_suffix}] calibrated {N_CALIB} OK ({time.time()-t0:.0f}s)", flush=True)
    converted = convert_pt2e(prepared)
    exported = torch.export.export(converted, ex)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    buf = edge.to_executorch().buffer
    out = ASSETS / f"teacher_wavlm_broad_int8_{tag_suffix}.pte"
    out.write_bytes(buf)
    print(f"  [{tag_suffix}] INT8 .pte -> {out} ({len(buf)/1e6:.0f} MB)", flush=True)
    return out


def main():
    t0 = time.time()
    _ensure_flatc()
    eager, deploy, tag = _load_and_merge("wavlm", "broad")
    deploy = deploy.eval()
    fp32_params = sum(p.numel() for p in deploy.parameters())
    print(f"[{tag}] merged, {fp32_params/1e6:.0f}M params ({time.time()-t0:.0f}s)",
          flush=True)

    waves = load_waves()
    tr, va, _y = _split_idx("broad")
    g = torch.Generator().manual_seed(2024)
    calib = waves[tr[torch.randperm(len(tr), generator=g)[:N_CALIB]]].contiguous()
    ex = (calib[:1].clone().contiguous(),)

    ladder = [
        (RELPOS, "relpos_excl"),                 # A: max coverage
        (RELPOS + ATTN, "ffn_only"),             # B: FFN-only fallback
    ]
    out = None
    for exclude, suffix in ladder:
        print(f"[attempt] exclude={exclude}", flush=True)
        try:
            out = build(deploy, calib, ex, exclude, suffix, t0)
            break
        except Exception as e:
            print(f"  [{suffix}] FAILED: {repr(e)[:300]}", flush=True)
    if out is None:
        print("ALL_ATTEMPTS_FAILED", flush=True)
        return

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
    mb = out.stat().st_size / 1e6
    print(f"[{tag}] WINNER={out.name} size={mb:.0f}MB "
          f"INT8-vs-fp32 max|Δ|={dmax:.3f} decision-agree={agree}/8 "
          f"({time.time()-t0:.0f}s)", flush=True)
    print("OK_WAVLM_INT8_SURGICAL", flush=True)


if __name__ == "__main__":
    main()
