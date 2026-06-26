"""Host benchmark + A/B harness for StressNet.

Measures, on the host CPU, the three things M11 needs as evidence — **latency**,
**model size**, and **accuracy** — for one or more StressNet variants, and emits
both a machine-readable JSON record and a human-readable markdown table.

This is the XNNPACK-CPU baseline. The QNN/Hexagon-NPU numbers (the "40% evidence")
come later through Qualcomm AI Hub (model/aihub_profile.py) and get compared
against the rows produced here. Nothing in this module touches the live AI Hub
token or a device, so it is safe to run unattended in the autonomy loop.

    python -m model.benchmark                       # default A/B suite -> docs/benchmarks/
    python -m model.benchmark --iters 50 --eval-n 64

Each variant flows through the real stack:
    StressNet(channels) --export_to_pte--> .pte --run_pte--> output
so the size and latency are of the actual program that ships, not a proxy.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .data import synthetic_dataset
from .export_executorch import export_to_pte
from .model import StressNet, example_input
from .run_pte import run_pte

Channels = tuple[int, int, int]


@dataclass
class BenchmarkResult:
    """One variant's measured latency / size / accuracy on the host."""

    name: str
    channels: Channels
    eager_latency_ms: float
    pte_latency_ms: float
    pt_bytes: int
    pte_bytes: int
    accuracy: float
    pte_max_abs_err: float
    eval_n: int
    iters: int

    @property
    def speedup_eager_over_pte(self) -> float:
        """>1 means the .pte runtime is faster than eager PyTorch."""
        return self.eager_latency_ms / self.pte_latency_ms if self.pte_latency_ms else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["speedup_eager_over_pte"] = round(self.speedup_eager_over_pte, 3)
        return d


@dataclass
class SuiteResult:
    results: list[BenchmarkResult]
    host: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"host": self.host, "results": [r.to_dict() for r in self.results]}


def _host_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }


def _time_ms(fn, *, warmup: int, iters: int) -> float:
    """Mean wall-clock milliseconds per call over ``iters`` timed runs."""
    for _ in range(max(0, warmup)):
        fn()
    iters = max(1, iters)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def _pt_size_bytes(model: torch.nn.Module) -> int:
    """Serialized checkpoint size, measured the way train.py saves it."""
    buf = io.BytesIO()
    torch.save({"model": model.state_dict()}, buf)
    return buf.getbuffer().nbytes


def benchmark_variant(
    name: str,
    *,
    channels: Channels = (16, 32, 64),
    warmup: int = 3,
    iters: int = 20,
    eval_n: int = 32,
    seed: int = 0,
) -> BenchmarkResult:
    """Measure latency, size, accuracy and eager/pte parity for one variant."""
    torch.manual_seed(seed)
    model = StressNet(channels=channels).eval()

    # --- accuracy on a held-out synthetic split ---
    x, y = synthetic_dataset(eval_n, seed=seed + 1000)
    with torch.no_grad():
        pred = model(x)
    accuracy = float(((pred >= 0.5).float() == y).float().mean())

    # --- eager latency (single representative window) ---
    one = example_input()
    with torch.no_grad():
        eager_ms = _time_ms(lambda: model(one), warmup=warmup, iters=iters)

    # --- export the real program; size is len(buffer) ---
    buffer = export_to_pte(model=model)
    pte_bytes = len(buffer)
    pt_bytes = _pt_size_bytes(model)

    with tempfile.TemporaryDirectory() as td:
        pte_path = Path(td) / f"{name}.pte"
        pte_path.write_bytes(buffer)

        # --- pte latency through the ExecuTorch runtime ---
        pte_ms = _time_ms(lambda: run_pte(pte_path, one), warmup=warmup, iters=iters)

        # --- parity: pte vs eager over a few eval inputs ---
        n_parity = min(eval_n, 8)
        max_err = 0.0
        with torch.no_grad():
            for i in range(n_parity):
                out = run_pte(pte_path, x[i:i + 1])
                max_err = max(max_err, float((out - pred[i:i + 1]).abs().max()))

    return BenchmarkResult(
        name=name,
        channels=channels,
        eager_latency_ms=round(eager_ms, 4),
        pte_latency_ms=round(pte_ms, 4),
        pt_bytes=pt_bytes,
        pte_bytes=pte_bytes,
        accuracy=round(accuracy, 4),
        pte_max_abs_err=max_err,
        eval_n=eval_n,
        iters=iters,
    )


def to_markdown(results: list[BenchmarkResult]) -> str:
    """Render an A/B comparison table, one row per variant."""
    header = (
        "| variant | channels | eager latency (ms) | pte latency (ms) | "
        "speedup | .pt size (KB) | .pte size (KB) | accuracy | parity max abs err |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r.name} | {tuple(r.channels)} | {r.eager_latency_ms:.3f} | "
            f"{r.pte_latency_ms:.3f} | {r.speedup_eager_over_pte:.2f}× | "
            f"{r.pt_bytes / 1024:.1f} | {r.pte_bytes / 1024:.1f} | "
            f"{r.accuracy:.3f} | {r.pte_max_abs_err:.2e} |"
        )
    return header + "\n".join(rows) + "\n"


def run_suite(
    variants: dict[str, Channels],
    *,
    out_dir: str | Path,
    warmup: int = 3,
    iters: int = 20,
    eval_n: int = 32,
    seed: int = 0,
) -> SuiteResult:
    """Benchmark each variant and write JSON + markdown artifacts to ``out_dir``."""
    results = [
        benchmark_variant(
            name, channels=ch, warmup=warmup, iters=iters, eval_n=eval_n, seed=seed
        )
        for name, ch in variants.items()
    ]
    suite = SuiteResult(results=results, host=_host_info())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(suite.to_dict(), indent=2) + "\n")

    md = (
        "# StressNet host benchmarks (XNNPACK-CPU baseline)\n\n"
        f"- host: `{suite.host['platform']}`\n"
        f"- torch: `{suite.host['torch']}`  · threads: {suite.host['threads']}\n\n"
        + to_markdown(results)
        + "\n_Latency is mean wall-clock per 1×[1,1,64,301] window. "
        "The NPU numbers (AI Hub) are compared against these CPU rows._\n"
        "\n_Accuracy reflects the weights benchmarked; variants are random-init "
        "by default, so 0.5 = chance. Latency, size, and eager↔pte parity are "
        "weight-independent and meaningful as shown._\n"
    )
    (out_dir / "benchmark.md").write_text(md)
    return suite


# The default A/B suite: a couple of width points spanning the size/accuracy curve.
DEFAULT_VARIANTS: dict[str, Channels] = {
    "small": (8, 16, 32),
    "base": (16, 32, 64),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark StressNet variants on the host")
    ap.add_argument("--out-dir", default="docs/benchmarks", help="artifact output dir")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    suite = run_suite(
        DEFAULT_VARIANTS,
        out_dir=args.out_dir,
        warmup=args.warmup,
        iters=args.iters,
        eval_n=args.eval_n,
        seed=args.seed,
    )
    print(to_markdown(suite.results))
    print(f"wrote {args.out_dir}/benchmark.json and benchmark.md")


if __name__ == "__main__":
    main()
