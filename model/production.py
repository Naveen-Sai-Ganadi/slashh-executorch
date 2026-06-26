"""Production recipe — the smallest *robust* StressNet, ready to ship.

The autonomy loop's experiments converged on one answer:

  * the trained width A/B (``model/ab_experiment.py``) recommended the smallest
    variant, ``tiny`` = ``(4, 8, 16)``; and
  * the robustness sweep (``model/robustness.py`` + ``model/robust_train.py``)
    showed that this tiny width, when trained with noise augmentation, holds
    accuracy through -5 dB SNR — far past where the clean-trained model collapses
    toward chance — at a fraction of the parameters. (The very-aggressive recipe
    ``{clean,20,10,5,0,-5,-10}`` was confirmed by the recipe-parameterized
    cross-init envelope, ``model/recipe_envelope.py``, to hold a -5 dB *cross-init*
    floor across 5 inits; going deeper (-15 dB training) does not move it, so -5 dB
    is the floor of the augmentation approach and the claim we stand on.)

So the production model is not a compromise between size and robustness: the
smallest net keeps the robustness that matters. This module bakes that into a
single recipe — build the tiny net, train it on a noise-augmented set, measure
its robustness curve, and export a ``.pte`` — and records the evidence under
``docs/benchmarks/`` so the choice is auditable.

    python -m model.production                       # train, measure, record
    python -m model.production --out assets/stress_model.pte   # also overwrite the shipped .pte

Non-destructive by default: it writes a benchmark record and leaves the
committed ``assets/`` artifacts alone unless ``--out`` is given. Host-only — no
device, no AI Hub, no live token.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import synthetic_dataset
from .export_executorch import export_quantized_to_pte, export_to_pte
from .model import StressNet
from .robust_train import noise_augmented_dataset
from .robustness import RobustnessResult, robustness_curve
from .run_pte import run_pte
from .train import train

# The A/B-recommended and most-robust width (see module docstring).
PRODUCTION_CHANNELS: tuple[int, int, int] = (4, 8, 16)


def build_production_model() -> StressNet:
    """The production architecture: the smallest, most-robust StressNet width."""
    return StressNet(channels=PRODUCTION_CHANNELS)


def train_production(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    eval_n_per_class: int = 96,
    eval_seeds: Sequence[int] | None = None,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.8,
) -> tuple[StressNet, dict]:
    """Train the production net on noise-augmented data and measure robustness.

    Returns the trained model and a meta dict carrying clean accuracy, the
    architecture, and the full robustness curve (so callers don't re-run it).
    With ``eval_seeds`` the robustness curve is averaged over several eval seeds
    (mean ± std); ``None`` uses the single seed ``seed + 1``.
    """
    # Seed before constructing the net so weight init is reproducible and
    # independent of any prior global-RNG use (e.g. test ordering).
    torch.manual_seed(seed)
    model = build_production_model()
    aug = noise_augmented_dataset(n_per_class, seed=seed)
    _, meta = train(
        data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
        n_per_class=0, seed=seed, model=model, dataset=aug,
    )
    curve = robustness_curve(
        model, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=seed + 1, eval_seeds=eval_seeds, threshold=threshold,
    )
    meta = {
        **meta,
        "channels": list(PRODUCTION_CHANNELS),
        "params": int(sum(p.numel() for p in model.parameters())),
        "robustness": curve.to_dict(),
    }
    return model, meta


def quantize_production(model: StressNet, *, calib_n: int = 24, seed: int = 2024) -> bytes:
    """INT8 (PT2E + XNNPACK) export of the production model — the deployable form.

    Calibrated on a clean synthetic set. NOTE: for a net this small the program
    is dominated by fixed runtime overhead, so INT8 barely shrinks the ``.pte``
    (~1.05x, vs ~3x on the base width). Its value here is integer compute on the
    NPU and predictions that stay put — not size. Host-only, no AI Hub token.
    """
    calib, _ = synthetic_dataset(calib_n, seed=seed)
    return export_quantized_to_pte(model.eval(), calib)


def _int8_max_abs_diff(model: StressNet, int8_pte: bytes, *, n: int = 16, seed: int = 4242) -> float:
    """Largest |int8 runtime score - eager score| over a clean eval set."""
    xs, _ = synthetic_dataset(n, seed=seed)
    with torch.no_grad():
        eager = model(xs).flatten()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pte = Path(td) / "prod_int8.pte"
        pte.write_bytes(int8_pte)
        worst = 0.0
        for i in range(xs.shape[0]):
            s = float(run_pte(pte, xs[i : i + 1]).flatten()[0])
            worst = max(worst, abs(s - float(eager[i])))
    return worst


@dataclass(frozen=True)
class ProductionResult:
    meta: dict
    pte_bytes: int
    robustness: RobustnessResult
    int8_bytes: int | None = None
    int8_max_abs_diff: float | None = None

    def to_dict(self) -> dict:
        d = {
            "channels": self.meta["channels"],
            "params": self.meta["params"],
            "val_acc": self.meta["val_acc"],
            "pte_bytes": self.pte_bytes,
            "robustness": self.robustness.to_dict(),
        }
        if self.int8_bytes is not None:
            d["int8_bytes"] = self.int8_bytes
            d["int8_max_abs_diff"] = self.int8_max_abs_diff
        return d


def _reliable_label(snr_db: float | None) -> str:
    return "clean only" if snr_db is None else f"{snr_db:g} dB"


def _drops_label(floor_db: float | None) -> str:
    return "never (holds at all tested SNRs)" if floor_db is None else f"{floor_db:g} dB"


def to_markdown(out: ProductionResult) -> str:
    r = out.robustness
    header = (
        "# Production model — smallest robust StressNet\n\n"
        f"Architecture **`{tuple(out.meta['channels'])}`** "
        f"({out.meta['params']:,} params), noise-augmented training, exported to "
        f"a **{out.pte_bytes / 1024:.1f} KB** `.pte`.\n\n"
        f"- clean val accuracy: **{out.meta['val_acc']:.3f}**\n"
        f"- reliable down to: **{_reliable_label(r.reliable_floor_db)}** "
        "(accuracy ≥ 0.80)\n"
        f"- drops below 0.80 at: {_drops_label(r.floor_db)}\n"
    )
    if out.int8_bytes is not None:
        header += (
            f"- **INT8** (PT2E + XNNPACK) export: **{out.int8_bytes / 1024:.1f} KB** "
            f"(scores within {out.int8_max_abs_diff:.4f} of eager). At ~"
            f"{out.meta['params']:,} params the program is overhead-dominated, so "
            "INT8's win here is integer compute on the NPU, not size.\n"
        )
    header += (
        "\nThe floors above are measured for *this* trained artifact. The "
        "conservative claim we stand on across re-trains is **-5 dB** — the "
        "cross-init reliable floor the very-aggressive training recipe holds across "
        "5 independent inits (see `model/recipe_envelope.py` and the project "
        "README).\n"
    )
    header += "\n| SNR | accuracy | f1 |\n|---|---|---|\n"
    rows = []
    for p in r.points:
        snr = "clean" if p.snr_db is None else f"{p.snr_db:g} dB"
        acc = f"{p.accuracy:.3f}"
        if p.acc_std is not None:
            acc += f" ± {p.acc_std:.3f}"
        rows.append(f"| {snr} | {acc} | {p.f1:.3f} |")
    return header + "\n".join(rows) + "\n"


def build_production(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    n_eval_seeds: int = 5,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.8,
    quantize: bool = True,
    out_dir: str | Path | None = "docs/benchmarks",
    pte_out: str | Path | None = None,
) -> ProductionResult:
    """Train, measure, export, and record the production model.

    Writes ``production.{json,md}`` into ``out_dir``. If ``pte_out`` is given,
    also writes the exported ``.pte`` there (overwriting the shipped artifact).
    With ``quantize`` (default), also produces the INT8 variant and records its
    size and its score agreement with the eager model.

    The shipped robustness record is averaged over ``n_eval_seeds`` eval seeds
    (mean ± std per SNR), so the floors it advertises carry a spread rather than
    resting on a single lucky draw. Set ``n_eval_seeds=1`` for the fast
    single-seed path (no std).
    """
    eval_seeds = (
        tuple(seed + 1 + i for i in range(n_eval_seeds)) if n_eval_seeds > 1 else None
    )
    model, meta = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        eval_seeds=eval_seeds, snr_levels=snr_levels, threshold=threshold,
    )
    pte = export_to_pte(model=model)
    int8_bytes = int8_diff = None
    if quantize:
        q = quantize_production(model)
        int8_bytes = len(q)
        int8_diff = _int8_max_abs_diff(model, q)
    result = ProductionResult(
        meta=meta,
        pte_bytes=len(pte),
        robustness=_rebuild_curve(meta["robustness"]),
        int8_bytes=int8_bytes,
        int8_max_abs_diff=int8_diff,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "production.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        (out_dir / "production.md").write_text(to_markdown(result))

    if pte_out is not None:
        pte_out = Path(pte_out)
        pte_out.parent.mkdir(parents=True, exist_ok=True)
        pte_out.write_bytes(pte)

    return result


def _rebuild_curve(d: dict) -> RobustnessResult:
    """Reconstruct a RobustnessResult from its serialized dict."""
    from .robustness import RobustnessPoint

    points = [
        RobustnessPoint(
            snr_db=p["snr_db"], accuracy=p["accuracy"], f1=p["f1"], n=p["n"],
            acc_std=p.get("acc_std"),
        )
        for p in d["points"]
    ]
    return RobustnessResult(
        points=points,
        floor_db=d["floor_db"],
        reliable_floor_db=d.get("reliable_floor_db"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the production StressNet")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--out", default=None, help="also write the .pte here (overwrites shipped artifact)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_production(
        epochs=args.epochs, n_per_class=args.n_per_class, seed=args.seed,
        out_dir=args.out_dir, pte_out=args.out,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/production.json and production.md")
    if args.out:
        print(f"wrote {args.out} ({out.pte_bytes:,} bytes)")


if __name__ == "__main__":
    main()
