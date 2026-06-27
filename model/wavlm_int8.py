"""Surgical INT8 (w8a8) quantization of the WavLM teacher (backlog #17).

The LoRA-merged ``microsoft/wavlm-large`` + stress head is ~317M params; in
fp32 its ExecuTorch ``.pte`` is ~1.2 GB. This module shrinks it to an INT8
``.pte`` (~317 MB) for the host/XNNPACK runtime by quantizing every Linear
*except* the relative-position attention path that breaks PT2E.

Why "surgical". A *global* PT2E pass crashes during calibration with
``Tensor dtype mismatch! Expected: Long, Got: float`` at
``scaled_dot_product_attention``: observers land on WavLM's gated
relative-position bias path (``gru_rel_pos_linear`` -> bias -> SDPA), and
``prepare_pt2e`` unconditionally runs ``transform_for_annotation ->
_convert_scalars_to_attrs``, which rewrites the int scalar WavLM uses as a Long
bucket index into a *float* tensor attr — crashing the index op. That subgraph
is tiny (one Linear + one Embedding per layer); the param-heavy work
(q/k/v/out_proj + the feed-forward linears) is ordinary and quantizes fine.

The fix has three parts:

  1. :class:`SafeQuantizer` makes ``transform_for_annotation`` a no-op — we only
     quantize Linears, whose correctness never needs the scalar->attr rewrite.
  2. :func:`make_exclude_filter` rejects nodes inside the rel-pos modules so no
     observer is ever placed on the Long-index path.
  3. Per-tensor (not per-channel) config: WavLM's hand-rolled attention matmuls
     aren't XNNPACK-partitioned, so their weight-dequant stays in the graph, and
     ``dequantize_per_channel`` has no portable out-variant (serialization
     fails) while ``dequantize_per_tensor`` does.

    PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 -m model.wavlm_int8

Host-only: the INT8 program is the **XNNPACK/CPU** ExecuTorch runtime, run via
``run_pte`` exactly as on a CPU device. This is distinct from the QNN/HTP *NPU*
path (``teacher_wavlm_broad_qnn.pte``, FP16-on-HTP) proven separately on the S25
Hexagon — same model, different backend. No AI Hub, no live token.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from executorch.runtime import Runtime

ASSETS = Path("assets")
BENCH = Path("docs/benchmarks")

# Module-path substrings whose Linear nodes must stay fp32. The rel-pos pair is
# the crashing path; the attention projections are the FFN-only safety fallback.
RELPOS_MODULES = ("gru_rel_pos_linear", "rel_attn_embed")
ATTN_MODULES = ("attention.k_proj", "attention.q_proj", "attention.v_proj",
                "attention.out_proj")

__all__ = [
    "RELPOS_MODULES",
    "ATTN_MODULES",
    "node_module_paths",
    "make_exclude_filter",
    "SafeQuantizer",
    "quantize_int8",
    "parity_metrics",
    "TeacherPte",
    "WavlmInt8Result",
    "evaluate_int8",
    "build_wavlm_int8",
    "to_markdown",
]


# --- pure: which modules does a graph node belong to? -----------------------

def node_module_paths(node) -> list[str]:
    """Every module path in a node's ``nn_module_stack`` (outermost..innermost).

    fx stores ``{key: (path, type)}``; some passes leave a bare string. Returns
    ``[]`` for nodes with no stack (free ops) so callers can keep them by default.
    """
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return []
    return [v[0] if isinstance(v, (tuple, list)) else str(v) for v in stack.values()]


def make_exclude_filter(exclude_substrings: Sequence[str]):
    """A ``keep(node) -> bool`` predicate for ``set_filter_function``.

    XNNPACKQuantizer keeps only nodes where the filter returns True, so this
    rejects any node whose module path contains an excluded substring. Nodes
    with no module stack are kept (never silently dropped).
    """
    def keep(node) -> bool:
        paths = node_module_paths(node)
        return not any(s in p for p in paths for s in exclude_substrings)
    return keep


# SafeQuantizer subclasses XNNPACKQuantizer lazily so importing this module
# (e.g. for the pure helpers / tests) doesn't force the heavy executorch backend
# import until quantization is actually requested.
def _make_safe_quantizer_cls():
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
    )

    class SafeQuantizer(XNNPACKQuantizer):
        """XNNPACKQuantizer whose ``transform_for_annotation`` is a no-op.

        The default runs ``_convert_scalars_to_attrs``, which rewrites WavLM's
        int rel-pos bucket index to a float attr and crashes the Long-index op.
        We only quantize Linears, so the rewrite is unnecessary — skip it.
        """

        def transform_for_annotation(self, model):
            return model

    return SafeQuantizer


def __getattr__(name):
    # Module-level lazy attribute so ``from model.wavlm_int8 import SafeQuantizer``
    # works without paying the executorch backend import at module load.
    if name == "SafeQuantizer":
        return _make_safe_quantizer_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- quantize: model -> INT8 .pte bytes -------------------------------------

def quantize_int8(
    model: torch.nn.Module,
    calib: torch.Tensor,
    *,
    exclude: Sequence[str] = RELPOS_MODULES,
    per_channel: bool = False,
) -> bytes:
    """Quantize ``model`` to an INT8 XNNPACK ``.pte`` and return its bytes.

    ``calib`` is a ``[N, ...]`` batch of representative inputs; ``calib[:1]`` is
    the export example shape and all N rows drive PT2E observation. ``exclude``
    lists module-path substrings to keep in fp32 (default: the rel-pos path).
    """
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        get_symmetric_quantization_config,
    )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    from .export_executorch import _ensure_flatc

    _ensure_flatc()
    SafeQuantizer = _make_safe_quantizer_cls()

    model = model.eval()
    example = (calib[:1].clone().contiguous(),)

    captured = torch.export.export(model, example).module()
    quantizer = SafeQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=per_channel))
    quantizer.set_filter_function(make_exclude_filter(exclude))

    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for i in range(calib.shape[0]):
            prepared(calib[i:i + 1])
    converted = convert_pt2e(prepared)

    exported = torch.export.export(converted, example)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    return bytes(edge.to_executorch().buffer)


# --- parity reduction -------------------------------------------------------

def parity_metrics(
    fp32_logits: torch.Tensor,
    int8_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
) -> dict:
    """INT8-vs-fp32 logit parity over a batch; optional accuracy vs labels.

    The binary decision is ``logit > 0``. Returns sample count, decision
    agreement (count + rate), and max/mean ``|Δlogit|``. With ``labels`` (0/1),
    adds each model's accuracy so the question "does INT8 keep the model's
    accuracy?" is answered directly — logit drift alone can be misleading.
    """
    f = fp32_logits.flatten().float()
    q = int8_logits.flatten().float()
    n = int(f.numel())
    fp, qp = f > 0, q > 0
    agree = int((fp == qp).sum().item())
    delta = (f - q).abs()
    out = {
        "n": n,
        "decision_agree": agree,
        "decision_agree_rate": round(agree / n, 4) if n else 0.0,
        "max_abs_delta": float(delta.max().item()) if n else 0.0,
        "mean_abs_delta": float(delta.mean().item()) if n else 0.0,
    }
    if labels is not None:
        y = labels.flatten().float()
        out["fp32_accuracy"] = round(float((fp.float() == y).float().mean().item()), 4)
        out["int8_accuracy"] = round(float((qp.float() == y).float().mean().item()), 4)
    return out


# --- runtime: load the INT8 .pte once, run it like the eager model ----------

class TeacherPte:
    """Load an INT8 ``.pte`` once and call it per-sample (fixed batch-1 input)."""

    def __init__(self, pte_bytes: bytes) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "wavlm_int8.pte"
        path.write_bytes(pte_bytes)
        program = Runtime.get().load_program(path)
        self._method = program.load_method("forward")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self._method.execute([x[i:i + 1]])[0] for i in range(x.shape[0])]
        return torch.cat(outs, dim=0)


@dataclass(frozen=True)
class WavlmInt8Result:
    tag: str
    variant: str
    fp32_params: int
    fp32_pte_mb: float | None
    int8_pte_mb: float
    n_eval: int
    metrics: dict
    calib_n: int
    eval_seconds: float

    def to_dict(self) -> dict:
        return {
            "model": self.tag,
            "task": "wavlm_int8",
            "backend": "xnnpack_cpu",
            "variant": self.variant,
            "dtype": "w8a8_per_tensor",
            "fp32_params": self.fp32_params,
            "fp32_pte_mb": self.fp32_pte_mb,
            "int8_pte_mb": round(self.int8_pte_mb, 1),
            "compression_x": (round(self.fp32_pte_mb / self.int8_pte_mb, 2)
                              if self.fp32_pte_mb else None),
            "n_eval": self.n_eval,
            "calib_n": self.calib_n,
            "eval_seconds": round(self.eval_seconds, 1),
            **self.metrics,
        }


def evaluate_int8(
    deploy: torch.nn.Module,
    pte_bytes: bytes,
    waves: torch.Tensor,
    idx: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> dict:
    """Run identical val waves through eager fp32 and the INT8 ``.pte``; compare.

    The INT8 program is loaded once (``TeacherPte``) and both models see the
    same inputs, so the parity is paired. ``labels`` (per-row 0/1) yields each
    model's accuracy on the evaluated subset.
    """
    deploy = deploy.eval()
    pte = TeacherPte(pte_bytes)
    fp32_logits, int8_logits = [], []
    with torch.no_grad():
        for i in idx.tolist():
            x = waves[i:i + 1].contiguous()
            fp32_logits.append(deploy(x).flatten())
            int8_logits.append(pte(x).flatten())
    f = torch.cat(fp32_logits)
    q = torch.cat(int8_logits)
    y = labels[idx] if labels is not None else None
    return parity_metrics(f, q, labels=y)


# --- orchestration: merge -> quantize -> evaluate -> record -----------------

def build_wavlm_int8(
    *,
    backbone: str = "wavlm",
    mapping: str = "broad",
    calib_n: int = 16,
    eval_n: int = 96,
    seed: int = 2024,
    out_dir: str | Path | None = "docs/benchmarks",
) -> WavlmInt8Result:
    """Full pipeline: load+merge the LoRA teacher, INT8-quantize it, evaluate
    paired parity/accuracy on held-out val speakers, and record the evidence."""
    from .export_teacher import _load_and_merge
    from .train_lora import load_waves, _split_idx

    t0 = time.time()
    _eager, deploy, tag = _load_and_merge(backbone, mapping)
    deploy = deploy.eval()
    fp32_params = sum(p.numel() for p in deploy.parameters())
    print(f"[{tag}] merged {fp32_params/1e6:.0f}M params ({time.time()-t0:.0f}s)", flush=True)

    waves = load_waves()
    tr, va, y = _split_idx(mapping)
    g = torch.Generator().manual_seed(seed)
    calib = waves[tr[torch.randperm(len(tr), generator=g)[:calib_n]]].contiguous()

    pte = quantize_int8(deploy, calib, exclude=RELPOS_MODULES)
    int8_mb = len(pte) / 1e6
    out_pte = ASSETS / f"teacher_{tag}_int8_relpos_excl.pte"
    out_pte.write_bytes(pte)
    print(f"[{tag}] INT8 .pte -> {out_pte} ({int8_mb:.0f} MB) ({time.time()-t0:.0f}s)",
          flush=True)

    # Held-out val subset: balance classes when possible for an honest accuracy.
    eval_idx = _balanced_subset(va, y, eval_n, g)
    metrics = evaluate_int8(deploy, pte, waves, eval_idx, labels=y)
    print(f"[{tag}] eval n={metrics['n']} "
          f"decision-agree={metrics['decision_agree']}/{metrics['n']} "
          f"fp32_acc={metrics.get('fp32_accuracy')} int8_acc={metrics.get('int8_accuracy')} "
          f"max|Δ|={metrics['max_abs_delta']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    result = WavlmInt8Result(
        tag=tag, variant="relpos_excl", fp32_params=fp32_params,
        fp32_pte_mb=round(fp32_params * 4 / 1e6, 1),  # fp32 weights ≈ 4 B/param
        int8_pte_mb=int8_mb, n_eval=metrics["n"], metrics=metrics,
        calib_n=calib_n, eval_seconds=time.time() - t0,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wavlm_int8.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        (out_dir / "wavlm_int8.md").write_text(to_markdown(result))
    return result


def _balanced_subset(idx: torch.Tensor, y: torch.Tensor, n: int,
                     gen: torch.Generator) -> torch.Tensor:
    """Up to ``n`` indices from ``idx``, class-balanced on ``y`` when possible."""
    labels = y[idx]
    per = max(1, n // 2)
    picks = []
    for c in (0.0, 1.0):
        pool = idx[labels == c]
        if len(pool) == 0:
            continue
        take = min(per, len(pool))
        picks.append(pool[torch.randperm(len(pool), generator=gen)[:take]])
    return torch.cat(picks) if picks else idx[:n]


_CHANCE = 0.5  # balanced binary task: a coin flip scores 0.5


def _verdict(r: WavlmInt8Result) -> str:
    """An honest, tiered bottom line. The size win is always real; the question
    is whether INT8 keeps enough predictive quality to be deployable. Accuracy
    is judged against the 0.5 chance line, not just against fp32."""
    m = r.metrics
    comp = f"{r.fp32_pte_mb / r.int8_pte_mb:.1f}×" if r.fp32_pte_mb else "—"
    if "fp32_accuracy" not in m:
        rate = m["decision_agree_rate"]
        if rate >= 0.95:
            return (f"**Verdict: usable.** Decision agreement is {rate*100:.0f}% "
                    f"and the artifact is {comp} smaller (no labels, so accuracy "
                    f"impact is inferred from agreement only).")
        return (f"**Verdict: characterize before use.** Decision agreement is only "
                f"{rate*100:.0f}% and no labels were supplied, so the accuracy "
                f"impact is unmeasured.")
    fp, q = m["fp32_accuracy"], m["int8_accuracy"]
    drop = fp - q
    near_chance = q <= _CHANCE + 0.06
    if drop <= 0.05 and not near_chance:
        return (f"**Verdict: viable.** INT8 keeps balanced accuracy "
                f"({q:.3f} vs fp32 {fp:.3f}, Δ {q-fp:+.3f}) while shrinking the "
                f"artifact {comp}. The host/XNNPACK INT8 `.pte` is a usable size "
                f"optimization.")
    if near_chance or drop >= 0.15:
        return (
            f"**Verdict: not viable — documented negative result.** The {comp} "
            f"size win is real, but balanced accuracy collapses from {fp:.3f} to "
            f"**{q:.3f}** — only {q-_CHANCE:+.3f} above the {_CHANCE:.3f} chance "
            f"line (Δ {q-fp:+.3f} vs fp32). Decision agreement of "
            f"{m['decision_agree']}/{m['n']} ({m['decision_agree_rate']*100:.0f}%) "
            f"is near coin-flip, so the binary decision is **not** preserved in any "
            f"useful sense. Naive per-tensor w8a8 does not work for this "
            f"{r.fp32_params/1e6:.0f}M-param transformer; recovering accuracy would "
            f"need per-channel weights (blocked here by the missing portable "
            f"`dequantize_per_channel` out-variant) or quantization-aware training. "
            f"The deployable path for this model stays the QNN/HTP **NPU** artifact "
            f"(FP16-on-HTP), proven on the S25 Hexagon with |Δlogit| ≈ 0.09.")
    return (
        f"**Verdict: marginal.** INT8 is {comp} smaller but balanced accuracy "
        f"drops {q-fp:+.3f} (fp32 {fp:.3f} → INT8 {q:.3f}). Usable only where the "
        f"size win outweighs the accuracy cost; prefer the QNN/HTP NPU path "
        f"(FP16-on-HTP) when accuracy matters.")


def to_markdown(r: WavlmInt8Result) -> str:
    m = r.metrics
    comp = (f"{r.fp32_pte_mb / r.int8_pte_mb:.1f}×" if r.fp32_pte_mb else "—")
    acc_line = ""
    if "fp32_accuracy" in m:
        delta = m["int8_accuracy"] - m["fp32_accuracy"]
        acc_line = (
            f"- balanced-set accuracy: fp32 **{m['fp32_accuracy']:.3f}** vs "
            f"INT8 **{m['int8_accuracy']:.3f}** (Δ {delta:+.3f}; chance = "
            f"{_CHANCE:.3f})\n")
    # Supporting note on the magnitude drift; the verdict carries the bottom line.
    drift_note = (
        "The max |Δlogit| is **large**: per-tensor INT8 on this transformer "
        "shifts raw logit magnitudes substantially, so the INT8 scores are not "
        "usable as calibrated probabilities — and (see verdict) here the sign of "
        "the logit degrades too, not just its magnitude."
        if m["max_abs_delta"] >= 1.0 else
        "Logit drift is small; both magnitude and decision are preserved.")
    return (
        f"# WavLM teacher — INT8 (w8a8) size optimization\n\n"
        f"LoRA-merged `microsoft/wavlm-large` + stress head "
        f"({r.fp32_params/1e6:.0f}M params), quantized to an INT8 ExecuTorch "
        f"`.pte` for the **XNNPACK/CPU** runtime (variant `{r.variant}`: every "
        f"Linear except the relative-position path). Distinct from the QNN/HTP "
        f"*NPU* path (FP16-on-HTP), which is proven separately on the S25 "
        f"Hexagon.\n\n"
        f"- size: fp32 ≈ **{r.fp32_pte_mb:.0f} MB** → INT8 **{r.int8_pte_mb:.0f} "
        f"MB** ({comp} smaller)\n"
        f"- calibration: {r.calib_n} train forwards (per-tensor symmetric w8a8)\n"
        f"- eval: {m['n']} held-out val speakers, paired (identical inputs)\n"
        f"- decision agreement INT8↔fp32: **{m['decision_agree']}/{m['n']}** "
        f"({m['decision_agree_rate']*100:.0f}%)\n"
        f"- |Δlogit|: max **{m['max_abs_delta']:.3f}**, mean "
        f"**{m['mean_abs_delta']:.3f}**\n"
        f"{acc_line}\n"
        f"{_verdict(r)}\n\n"
        f"{drift_note}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Surgical INT8 for the WavLM teacher")
    ap.add_argument("--backbone", default="wavlm")
    ap.add_argument("--mapping", default="broad")
    ap.add_argument("--calib-n", type=int, default=16)
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--out-dir", default="docs/benchmarks")
    args = ap.parse_args()

    r = build_wavlm_int8(
        backbone=args.backbone, mapping=args.mapping,
        calib_n=args.calib_n, eval_n=args.eval_n, out_dir=args.out_dir,
    )
    print(to_markdown(r))
    print(f"wrote {args.out_dir}/wavlm_int8.json and wavlm_int8.md")


if __name__ == "__main__":
    main()
