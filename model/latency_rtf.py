"""End-to-end per-window latency + Real-Time Factor (RTF) benchmark.

The always-on product claim is *real-time, offline, on-device*: every
``HOP_SECONDS`` a new ``WINDOW_SECONDS`` audio window must be turned into a
stress score before the next hop arrives. That budget covers the **whole**
chain — raw PCM -> log-mel features -> model score — not just the model forward
pass that ``model/benchmark.py`` already times.

This harness times the full chain per window for a given backend (eager fp32,
or an exported ``.pte``), decomposes the cost into *feature extraction* vs
*inference*, and reports the Real-Time Factor:

    RTF = per-window wall-clock / audio duration

against two deadlines — the window length (how fast we chew through audio) and,
more importantly, the streaming **hop** (the latency budget for an always-on
detector). ``RTF < 1`` means real-time; ``RTF_hop`` is the tighter bound.

    python -m model.latency_rtf                  # train, time eager + fp32 + int8

Pure host measurement: no device, no AI Hub, no live token. The absolute
numbers are machine-dependent (they characterize *this* host's XNNPACK-CPU
path); the value is the decomposition and the head-room they expose, which a
faster NPU only improves on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from .audio_config import HOP_SECONDS, WINDOW_SAMPLES, WINDOW_SECONDS
from .export_executorch import export_to_pte, export_quantized_to_pte
from .features import extract
from .int8_robustness import PteModule
from .model import StressNet

__all__ = [
    "StageTiming",
    "LatencyResult",
    "LatencyReport",
    "measure_latency",
    "build_latency_rtf",
    "to_markdown",
    "main",
]


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # nearest-rank: index = ceil(q * N) - 1, clamped
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


@dataclass(frozen=True)
class StageTiming:
    """Summary stats (milliseconds) for one stage across the timed iterations."""

    name: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float

    @staticmethod
    def from_samples(name: str, samples_ms: Sequence[float]) -> "StageTiming":
        s = sorted(samples_ms)
        mean = sum(s) / len(s)
        return StageTiming(
            name=name,
            mean_ms=mean,
            p50_ms=_percentile(s, 0.50),
            p95_ms=_percentile(s, 0.95),
            min_ms=s[0],
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "min_ms": self.min_ms,
        }


@dataclass(frozen=True)
class LatencyResult:
    """Per-window latency for one backend, with derived Real-Time Factors."""

    backend: str
    stages: list[StageTiming]  # [feature_extraction, inference]
    total_mean_ms: float
    total_p95_ms: float
    window_seconds: float
    hop_seconds: float
    rtf_window: float
    rtf_hop: float
    real_time: bool
    iters: int

    @property
    def feature_extraction(self) -> StageTiming:
        return self.stages[0]

    @property
    def inference(self) -> StageTiming:
        return self.stages[1]

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "stages": [s.to_dict() for s in self.stages],
            "total_mean_ms": self.total_mean_ms,
            "total_p95_ms": self.total_p95_ms,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "rtf_window": self.rtf_window,
            "rtf_hop": self.rtf_hop,
            "real_time": self.real_time,
            "iters": self.iters,
        }


def _infer_fn(model: StressNet | None, pte_bytes: bytes | None):
    """Return (callable that scores a [1,1,N_MELS,N_FRAMES] tensor, backend label)."""
    if (model is None) == (pte_bytes is None):
        raise ValueError("pass exactly one of `model` or `pte_bytes`")
    if model is not None:
        model = model.eval()

        @torch.no_grad()
        def run(feat: torch.Tensor) -> torch.Tensor:
            return model(feat)

        return run, "eager"
    module = PteModule(pte_bytes)

    def run(feat: torch.Tensor) -> torch.Tensor:
        # The ExecuTorch runtime requires a contiguous (or channels-last) input;
        # extract()'s reshape can hand back a non-contiguous view.
        return module(feat.contiguous())

    return run, "pte"


def measure_latency(
    *,
    model: StressNet | None = None,
    pte_bytes: bytes | None = None,
    backend: str | None = None,
    iters: int = 50,
    warmup: int = 5,
    seed: int = 0,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> LatencyResult:
    """Time the full PCM -> log-mel -> score chain for one backend.

    Exactly one of ``model`` (eager fp32) or ``pte_bytes`` (an exported
    program) must be given. Each timed iteration extracts log-mel features from
    a fresh random window and scores them; the two stages are timed separately
    so the report shows where the budget goes. ``warmup`` untimed iterations
    prime caches/JIT before measurement. Returns a :class:`LatencyResult` whose
    RTFs are the per-window wall-clock over the window and hop durations.
    """
    run, default_label = _infer_fn(model, pte_bytes)
    label = backend if backend is not None else default_label

    gen = torch.Generator().manual_seed(seed)
    # Distinct windows per iteration (incl. warmup) — realistic, deterministic.
    windows = [
        torch.randn(WINDOW_SAMPLES, generator=gen) for _ in range(warmup + iters)
    ]

    # Warmup: exercise both stages, discard timings.
    for w in windows[:warmup]:
        run(extract(w))

    fe_ms: list[float] = []
    inf_ms: list[float] = []
    for w in windows[warmup:]:
        t0 = perf_counter()
        feat = extract(w)
        t1 = perf_counter()
        run(feat)
        t2 = perf_counter()
        fe_ms.append((t1 - t0) * 1000.0)
        inf_ms.append((t2 - t1) * 1000.0)

    fe = StageTiming.from_samples("feature_extraction", fe_ms)
    inf = StageTiming.from_samples("inference", inf_ms)
    total_mean = fe.mean_ms + inf.mean_ms
    # p95 of the per-iter totals (not the sum of per-stage p95s).
    totals = sorted(a + b for a, b in zip(fe_ms, inf_ms))
    total_p95 = _percentile(totals, 0.95)

    rtf_window = (total_mean / 1000.0) / window_seconds
    rtf_hop = (total_mean / 1000.0) / hop_seconds
    return LatencyResult(
        backend=label,
        stages=[fe, inf],
        total_mean_ms=total_mean,
        total_p95_ms=total_p95,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        rtf_window=rtf_window,
        rtf_hop=rtf_hop,
        real_time=rtf_hop < 1.0,
        iters=iters,
    )


@dataclass(frozen=True)
class LatencyReport:
    """Latency across several backends for the same model + audio config."""

    results: list[LatencyResult]
    window_seconds: float
    hop_seconds: float

    @property
    def real_time_all(self) -> bool:
        return all(r.real_time for r in self.results)

    @property
    def fastest(self) -> str:
        return min(self.results, key=lambda r: r.total_mean_ms).backend

    def to_dict(self) -> dict:
        return {
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "real_time_all": self.real_time_all,
            "fastest": self.fastest,
            "results": [r.to_dict() for r in self.results],
        }


def _rtf_label(rtf: float) -> str:
    return f"{rtf:.4g}× ({1.0 / rtf:.0f}× real-time head-room)" if rtf > 0 else "—"


def to_markdown(report: LatencyReport) -> str:
    verdict = (
        "**all backends clear real-time**"
        if report.real_time_all
        else "**some backend misses the hop budget**"
    )
    header = (
        "# End-to-end latency & Real-Time Factor\n\n"
        f"Per-window cost of the full **PCM → log-mel → score** chain on this "
        f"host (XNNPACK-CPU). Window **{report.window_seconds:g} s**, streaming "
        f"hop **{report.hop_seconds:g} s** — every hop a new window must be "
        f"scored before the next arrives. {verdict}; fastest backend: "
        f"**`{report.fastest}`**.\n\n"
        "RTF is per-window wall-clock ÷ audio duration; **RTF(hop)** is the one "
        "that must stay < 1 for an always-on detector.\n\n"
        "| backend | feature ms | inference ms | total ms (mean / p95) | "
        "RTF(window) | RTF(hop) | real-time? |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in report.results:
        rows.append(
            f"| `{r.backend}` | {r.feature_extraction.mean_ms:.3f} | "
            f"{r.inference.mean_ms:.3f} | {r.total_mean_ms:.3f} / {r.total_p95_ms:.3f} | "
            f"{r.rtf_window:.4g} | {r.rtf_hop:.4g} | "
            f"{'✅' if r.real_time else '❌'} |"
        )
    note = (
        "\n\nFeature extraction dominates the budget for a net this small, so "
        "the practical optimization lever is the log-mel front-end, not the "
        "model. The figures are this host's CPU path; the on-device NPU only "
        "widens the head-room.\n"
    )
    return header + "\n".join(rows) + note


def build_latency_rtf(
    *,
    iters: int = 100,
    warmup: int = 10,
    seed: int = 0,
    quantize: bool = True,
    out_dir: str | Path | None = "docs/benchmarks",
) -> LatencyReport:
    """Build the production net, time eager + fp32-`.pte` (+ INT8 `.pte`), record.

    Writes ``latency_rtf.{json,md}`` into ``out_dir``. The model is built but
    NOT trained — latency is weight-independent, so this stays fast and
    deterministic. With ``quantize`` (default) the INT8 program is timed too.
    """
    from .production import build_production_model
    from .data import synthetic_dataset

    model = build_production_model().eval()
    results = [measure_latency(model=model, backend="eager", iters=iters, warmup=warmup, seed=seed)]

    fp32_pte = export_to_pte(model=model)
    results.append(
        measure_latency(pte_bytes=fp32_pte, backend="pte-fp32", iters=iters, warmup=warmup, seed=seed)
    )

    if quantize:
        calib, _ = synthetic_dataset(24, seed=2024)
        int8_pte = export_quantized_to_pte(model, calib)
        results.append(
            measure_latency(pte_bytes=int8_pte, backend="pte-int8", iters=iters, warmup=warmup, seed=seed)
        )

    report = LatencyReport(
        results=results, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "latency_rtf.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        (out_dir / "latency_rtf.md").write_text(to_markdown(report))

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end latency + RTF benchmark")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    report = build_latency_rtf(
        iters=args.iters, warmup=args.warmup, quantize=not args.no_quantize,
        out_dir=args.out_dir,
    )
    print(to_markdown(report))
    print(f"wrote {args.out_dir}/latency_rtf.json and latency_rtf.md")


if __name__ == "__main__":
    main()
