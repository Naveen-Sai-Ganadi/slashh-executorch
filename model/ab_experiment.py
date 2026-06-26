"""Trained architecture A/B experiment for StressNet width variants.

The raw benchmark harness (model/benchmark.py) measures latency/size/parity on
**random-init** nets, so its accuracy column is chance (0.5) by design. This
module closes that gap: it briefly *trains* each width variant on the synthetic
task, then measures real validation accuracy alongside the exported program's
size and eager latency, and picks a recommended variant from the
accuracy-vs-size trade-off.

    python -m model.ab_experiment                 # default width sweep
    python -m model.ab_experiment --epochs 15 --n-per-class 96

Everything runs on the host through the real export stack
(StressNet --export_to_pte--> .pte). It does NOT touch Qualcomm AI Hub or the
live token (that is the gated M5); it is safe to run unattended in the loop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .benchmark import _time_ms
from .export_executorch import export_to_pte
from .model import StressNet, example_input
from .train import train

Channels = tuple[int, int, int]


@dataclass
class ABVariantResult:
    """One trained width variant: real accuracy + exported size + latency."""

    name: str
    channels: Channels
    val_acc: float
    params: int
    pte_bytes: int
    eager_latency_ms: float

    @property
    def acc_per_kb(self) -> float:
        """Validation accuracy bought per KB of program — the efficiency knob."""
        kb = self.pte_bytes / 1024
        return self.val_acc / kb if kb else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["acc_per_kb"] = round(self.acc_per_kb, 6)
        return d


@dataclass
class ABResult:
    results: list[ABVariantResult]
    recommended: str

    def to_dict(self) -> dict:
        return {
            "recommended": self.recommended,
            "results": [r.to_dict() for r in self.results],
        }


def _recommend(results: list[ABVariantResult]) -> str:
    """Pick the variant with the best accuracy, breaking ties toward smaller.

    Accuracy is the product goal; among variants within a hair of the best
    accuracy we prefer the smaller program (cheaper to ship to the NPU).
    """
    best_acc = max(r.val_acc for r in results)
    contenders = [r for r in results if best_acc - r.val_acc <= 0.02]
    return min(contenders, key=lambda r: r.pte_bytes).name


def run_ab(
    variants: dict[str, Channels],
    *,
    epochs: int = 8,
    n_per_class: int = 64,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> ABResult:
    """Train + measure each width variant; optionally write JSON + markdown.

    Each variant value is a ``(c1, c2, c3)`` channels tuple. Returns an
    ``ABResult`` with one ``ABVariantResult`` per variant and a recommendation.
    """
    results: list[ABVariantResult] = []
    one = example_input()

    for name, channels in variants.items():
        channels = tuple(channels)
        net = StressNet(channels=channels)
        _, meta = train(
            data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
            n_per_class=n_per_class, seed=seed, model=net,
        )
        net.eval()

        params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            eager_ms = _time_ms(lambda: net(one), warmup=2, iters=10)
        pte_bytes = len(export_to_pte(model=net))

        results.append(
            ABVariantResult(
                name=name,
                channels=channels,
                val_acc=round(float(meta["val_acc"]), 4),
                params=params,
                pte_bytes=pte_bytes,
                eager_latency_ms=round(eager_ms, 4),
            )
        )

    recommended = _recommend(results)
    out = ABResult(results=results, recommended=recommended)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ab_experiment.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "ab_experiment.md").write_text(to_markdown(out))

    return out


def to_markdown(out: ABResult) -> str:
    """Render a trained-A/B table, one row per variant, recommendation noted."""
    header = (
        "# StressNet trained architecture A/B\n\n"
        "Each width variant is trained briefly on the synthetic task, then "
        "exported. Accuracy is **real validation accuracy** (not chance), so "
        "the size-vs-accuracy trade-off is decision-grade.\n\n"
        "| variant | channels | params | val acc | .pte size (KB) | "
        "eager latency (ms) | acc / KB |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in out.results:
        star = " ⭐" if r.name == out.recommended else ""
        rows.append(
            f"| {r.name}{star} | {tuple(r.channels)} | {r.params:,} | "
            f"{r.val_acc:.3f} | {r.pte_bytes / 1024:.1f} | "
            f"{r.eager_latency_ms:.3f} | {r.acc_per_kb:.4f} |"
        )
    footer = (
        f"\n\n**Recommended: `{out.recommended}`** — best validation accuracy, "
        "ties broken toward the smaller program.\n"
    )
    return header + "\n".join(rows) + footer


# Default width sweep: a few points along the capacity curve.
DEFAULT_VARIANTS: dict[str, Channels] = {
    "tiny": (4, 8, 16),
    "small": (8, 16, 32),
    "base": (16, 32, 64),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Trained A/B over StressNet widths")
    ap.add_argument("--out-dir", default="docs/benchmarks", help="artifact output dir")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = run_ab(
        DEFAULT_VARIANTS,
        epochs=args.epochs,
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/ab_experiment.json and ab_experiment.md")


if __name__ == "__main__":
    main()
