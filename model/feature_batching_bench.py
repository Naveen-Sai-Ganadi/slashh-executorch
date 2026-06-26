"""Front-end throughput A/B: per-sample ``extract`` loop vs ``extract_batch``.

The latency benchmark (``model/latency_rtf.py``) found the log-mel front-end is
the dominant per-window cost, and every host-side dataset / robustness / A-B
builder pays it in a Python ``for`` loop — one ``MelSpectrogram`` call per
window, hundreds per run. ``model.features.extract_batch`` vectorizes that
front-end across the whole batch in a single call.

This harness measures the wall-clock to featurize ``N`` windows both ways and
reports the speedup, and asserts the two paths agree (the batched mel matmul
reorders reductions, so allclose, not bit-equal). It is a *host* throughput
characterization — the on-device extractor still streams one window at a time,
so this does not change device behavior; it speeds up training/eval/A-B builds
(and therefore the autonomy loop itself).

    python -m model.feature_batching_bench       # write docs/benchmarks/feature_batching.{json,md}

Absolute numbers are machine-dependent; the ratio is the point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from .audio_config import WINDOW_SAMPLES
from .features import extract, extract_batch

__all__ = [
    "BatchingResult",
    "measure_batching",
    "to_markdown",
    "build_feature_batching",
    "main",
]


@dataclass(frozen=True)
class BatchingResult:
    """Per-sample-loop vs batched front-end timing for ``n`` windows."""

    n: int
    repeats: int
    loop_ms: float          # best wall-clock to featurize n windows via extract loop
    batch_ms: float         # best wall-clock to featurize n windows via extract_batch
    speedup: float          # loop_ms / batch_ms
    max_abs_diff: float     # max |batched - per-sample| over the batch
    parity_atol: float
    parity_ok: bool

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "repeats": self.repeats,
            "loop_ms": self.loop_ms,
            "batch_ms": self.batch_ms,
            "speedup": self.speedup,
            "max_abs_diff": self.max_abs_diff,
            "parity_atol": self.parity_atol,
            "parity_ok": self.parity_ok,
        }


def measure_batching(
    *,
    n: int = 128,
    repeats: int = 5,
    warmup: int = 2,
    seed: int = 0,
    parity_atol: float = 1e-5,
) -> BatchingResult:
    """Time featurizing ``n`` random windows via the loop vs ``extract_batch``.

    Each of ``repeats`` timed trials featurizes the same ``n`` windows both
    ways; the *best* (min) wall-clock per path is reported to suppress noise.
    ``warmup`` untimed trials prime caches first. Parity is checked once on the
    featurized batch.
    """
    gen = torch.Generator().manual_seed(seed)
    batch = torch.stack([torch.randn(WINDOW_SAMPLES, generator=gen) for _ in range(n)])

    def run_loop() -> torch.Tensor:
        return torch.cat([extract(batch[i]) for i in range(n)], dim=0)

    def run_batch() -> torch.Tensor:
        return extract_batch(batch)

    for _ in range(warmup):
        run_loop()
        run_batch()

    loop_best = float("inf")
    batch_best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        loop_out = run_loop()
        t1 = perf_counter()
        batch_out = run_batch()
        t2 = perf_counter()
        loop_best = min(loop_best, (t1 - t0) * 1000.0)
        batch_best = min(batch_best, (t2 - t1) * 1000.0)

    max_abs_diff = float((batch_out - loop_out).abs().max())
    return BatchingResult(
        n=n,
        repeats=repeats,
        loop_ms=loop_best,
        batch_ms=batch_best,
        speedup=loop_best / batch_best if batch_best > 0 else float("nan"),
        max_abs_diff=max_abs_diff,
        parity_atol=parity_atol,
        parity_ok=max_abs_diff <= parity_atol,
    )


def to_markdown(r: BatchingResult) -> str:
    parity = "✅ parity" if r.parity_ok else "❌ PARITY BREAK"
    return (
        "# Front-end throughput: per-sample loop vs `extract_batch`\n\n"
        f"Wall-clock to featurize **{r.n}** windows on this host (best of "
        f"{r.repeats} trials). `extract_batch` runs the log-mel front-end as a "
        "single batched `MelSpectrogram` call instead of one call per window.\n\n"
        "| path | time (ms) | per-window (ms) |\n"
        "|---|---|---|\n"
        f"| per-sample `extract` loop | {r.loop_ms:.3f} | {r.loop_ms / r.n:.4f} |\n"
        f"| `extract_batch` | {r.batch_ms:.3f} | {r.batch_ms / r.n:.4f} |\n\n"
        f"**Speedup: {r.speedup:.2f}×** building {r.n} windows. "
        f"{parity} (max |Δ| {r.max_abs_diff:.2e} ≤ {r.parity_atol:.0e}).\n\n"
        "Host throughput only — the device extractor streams one window at a "
        "time and is unchanged. The win compounds across every dataset / "
        "robustness / A-B build (and the autonomy loop's own test suite).\n"
    )


def build_feature_batching(
    *,
    n: int = 128,
    repeats: int = 5,
    seed: int = 0,
    out_dir: str | Path | None = "docs/benchmarks",
) -> BatchingResult:
    """Measure the front-end speedup and write ``feature_batching.{json,md}``."""
    result = measure_batching(n=n, repeats=repeats, seed=seed)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feature_batching.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        (out_dir / "feature_batching.md").write_text(to_markdown(result))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Front-end batching throughput A/B")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    result = build_feature_batching(n=args.n, repeats=args.repeats, out_dir=args.out_dir)
    print(to_markdown(result))
    print(f"wrote {args.out_dir}/feature_batching.json and feature_batching.md")


if __name__ == "__main__":
    main()
