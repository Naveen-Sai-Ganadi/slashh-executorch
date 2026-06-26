"""Cross-initialization robustness envelope for the shipped StressNet.

The README stands on a specific, falsifiable claim: the production net is small
enough (~1,549 params) that its noise floor is *init-sensitive*, so "0 dB is
the envelope we stand on." That is a statement about variance across random
**initializations** — and nothing measured it. ``model/robustness.py``'s
``eval_seeds`` averaging varies the *eval draw* with the model held fixed; it
says nothing about how much the floor moves when you reroll the weights.

This harness trains several independent inits, runs each through the same noise
sweep, and reports:

* the per-init reliable floor (the SNR each model is reliable down to), and
* the **conservative envelope** — the *worst* (least-noisy) of those floors, the
  SNR every init still clears regardless of which one training happens to land.

That envelope is the honest number to advertise: it is what holds no matter the
seed, not the floor of a lucky init.

    python -m model.init_envelope --n-inits 5

Records ``docs/benchmarks/init_envelope.{json,md}``. Host-only; no device, no
AI Hub token.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import StressNet
from .robustness import _snr_label, robustness_curve

__all__ = [
    "InitFloor",
    "InitEnvelopeResult",
    "init_envelope",
    "build_init_envelope",
]


@dataclass(frozen=True)
class InitFloor:
    """One initialization's clean accuracy and reliable noise floor."""

    seed: int
    clean_acc: float
    reliable_floor_db: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InitEnvelopeResult:
    floors: list[InitFloor]
    threshold: float
    # the SNR every init is still reliable down to — max of the per-init floors,
    # since reliable_floor_db is the lowest SNR each model still clears. None if
    # any init is reliable only on clean audio (no common noisy floor).
    envelope_db: float | None

    @property
    def floor_values(self) -> list[float]:
        return [f.reliable_floor_db for f in self.floors if f.reliable_floor_db is not None]

    def to_dict(self) -> dict:
        vals = self.floor_values
        return {
            "threshold": self.threshold,
            "envelope_db": self.envelope_db,
            "n_inits": len(self.floors),
            "floor_min_db": min(vals) if vals else None,
            "floor_median_db": statistics.median(vals) if vals else None,
            "floor_max_db": max(vals) if vals else None,
            "floors": [f.to_dict() for f in self.floors],
        }


def _envelope(floors: list[InitFloor]) -> float | None:
    vals = [f.reliable_floor_db for f in floors]
    if any(v is None for v in vals):
        return None  # some init holds only on clean audio -> no common noisy floor
    return max(vals) if vals else None


def init_envelope(
    models: Sequence[tuple[int, StressNet]],
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_per_class: int = 64,
    eval_seed: int = 1,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> InitEnvelopeResult:
    """Reduce a set of trained inits to a conservative robustness envelope.

    ``models`` is a sequence of ``(seed, model)`` pairs (the seed is recorded
    for provenance only). Every init is evaluated on the *same* eval draw
    (``eval_seed``) so differences reflect the weights, not the noise sample.
    """
    floors: list[InitFloor] = []
    for seed, model in models:
        curve = robustness_curve(
            model.eval(), snr_levels=snr_levels, n_per_class=n_per_class,
            seed=eval_seed, threshold=threshold,
        )
        clean = next((p.accuracy for p in curve.points if p.snr_db is None), float("nan"))
        floors.append(
            InitFloor(
                seed=seed,
                clean_acc=round(clean, 4),
                reliable_floor_db=curve.reliable_floor_db,
            )
        )

    out = InitEnvelopeResult(
        floors=floors, threshold=threshold, envelope_db=_envelope(floors)
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "init_envelope.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "init_envelope.md").write_text(to_markdown(out))

    return out


def _floor_label(snr_db: float | None) -> str:
    return "clean only" if snr_db is None else _snr_label(snr_db)


def to_markdown(out: InitEnvelopeResult) -> str:
    if out.envelope_db is None:
        verdict = (
            "**No common noisy envelope** — at least one initialization is "
            "reliable only on clean audio, so the floor genuinely depends on the "
            "draw. Advertise robustness on clean audio and lean on smoothing "
            "under noise."
        )
    else:
        verdict = (
            f"**Conservative envelope: reliable down to {_snr_label(out.envelope_db)} "
            f"across every initialization** (threshold {out.threshold:.2f}). This is "
            "the floor that holds regardless of the training seed — the honest "
            "number to stand on."
        )
    d = out.to_dict()
    spread = (
        f"{_floor_label(d['floor_min_db'])} … {_floor_label(d['floor_max_db'])} "
        f"(median {_floor_label(d['floor_median_db'])})"
        if d["floor_min_db"] is not None
        else "n/a (no init reliable below clean)"
    )
    header = (
        "# Cross-initialization robustness envelope\n\n"
        f"{len(out.floors)} independent initializations of the shipped width were "
        "trained and run through the same noise sweep (same eval draw). The "
        "**envelope** is the least-noisy per-init floor — the SNR every init "
        "still clears.\n\n"
        f"- per-init reliable floor: {spread}\n"
        f"- {verdict}\n\n"
        "| init (seed) | clean acc | reliable down to |\n"
        "|---|---|---|\n"
    )
    rows = [
        f"| {f.seed} | {f.clean_acc:.3f} | {_floor_label(f.reliable_floor_db)} |"
        for f in out.floors
    ]
    return header + "\n".join(rows) + "\n"


def build_init_envelope(
    *,
    n_inits: int = 5,
    epochs: int = 12,
    n_per_class: int = 96,
    base_seed: int = 0,
    eval_n_per_class: int = 64,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> InitEnvelopeResult:
    """Train ``n_inits`` production models on distinct seeds; derive the envelope."""
    from .production import train_production

    models: list[tuple[int, StressNet]] = []
    for i in range(n_inits):
        seed = base_seed + i
        model, _ = train_production(
            epochs=epochs, n_per_class=n_per_class, seed=seed,
            snr_levels=snr_levels, threshold=threshold,
        )
        models.append((seed, model))
    return init_envelope(
        models, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        eval_seed=base_seed + 1000, threshold=threshold, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-init robustness envelope")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--n-inits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    out = build_init_envelope(
        n_inits=args.n_inits, epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, base_seed=args.base_seed,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/init_envelope.json and init_envelope.md")


if __name__ == "__main__":
    main()
