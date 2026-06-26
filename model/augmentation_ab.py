"""A/B the noise-augmentation *recipe* — which training-time SNRs buy the
deepest reliable floor?

The arc that leads here:

* ``robustness.py`` — the clean-trained net collapses by ~10 dB.
* ``robust_train.py`` — one noise-augmentation recipe fixes that, raising the
  floor. It proves augmentation helps, but with a *single* fixed recipe.
* ``detector_noise_ab.py`` — detector knobs can't push the 10 dB floor deeper;
  the lever is training, not the detector.

So the open question is *how much* noise to train on. Augment too gently and
the floor stays shallow; augment with very low SNRs and you may corrupt the
clean-audio accuracy you actually ship on. This harness sweeps several recipes
(each a set of training-time SNR choices) over the *same* architecture and the
*same* eval draw, then recommends the recipe with the **deepest reliable floor
that doesn't sacrifice clean accuracy** — turning "augmentation helps" into a
concrete training recommendation.

    python -m model.augmentation_ab

Records ``docs/benchmarks/augmentation_ab.{json,md}``. Host-only; no device,
no AI Hub token.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import StressNet
from .robust_train import noise_augmented_dataset
from .robustness import RobustnessResult, robustness_curve
from .train import train

__all__ = [
    "AugRecipe",
    "RecipeResult",
    "AugmentationABResult",
    "augmentation_ab",
    "build_augmentation_ab",
]


@dataclass(frozen=True)
class AugRecipe:
    """One training-augmentation recipe: the SNRs mixed in during training.

    ``snr_choices`` is the per-window draw pool; ``None`` in the pool means
    "leave this window clean". ``(None,)`` is clean-only training.
    """

    label: str
    snr_choices: tuple[float | None, ...]

    def to_dict(self) -> dict:
        return {"label": self.label, "snr_choices": list(self.snr_choices)}


@dataclass(frozen=True)
class RecipeResult:
    recipe: AugRecipe
    clean_acc: float                   # accuracy on the clean eval point
    reliable_floor_db: float | None    # deepest SNR still ≥ threshold (None = none)
    points: list[dict]                 # per-SNR accuracy rows

    def to_dict(self) -> dict:
        return {
            "recipe": self.recipe.to_dict(),
            "clean_acc": self.clean_acc,
            "reliable_floor_db": self.reliable_floor_db,
            "points": self.points,
        }


@dataclass(frozen=True)
class AugmentationABResult:
    results: list[RecipeResult]
    recommended: RecipeResult | None
    threshold: float
    min_clean_acc: float

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "min_clean_acc": self.min_clean_acc,
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "results": [r.to_dict() for r in self.results],
        }


# A recipe with no reliable noisy floor must sort *worse* than any real floor.
_NO_FLOOR = float("inf")


def _clean_acc(curve: RobustnessResult) -> float:
    for p in curve.points:
        if p.snr_db is None:
            return p.accuracy
    return float("nan")


def _evaluate(
    recipe: AugRecipe, *, snr_levels, n_per_class: int, eval_n_per_class: int,
    epochs: int, batch_size: int, lr: float, seed: int, threshold: float,
) -> RecipeResult:
    model = StressNet(channels=(4, 8, 16))
    if recipe.snr_choices == (None,):
        # clean-only: train on the plain synthetic set
        train(data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
              n_per_class=n_per_class, seed=seed, model=model)
    else:
        data = noise_augmented_dataset(
            n_per_class, seed=seed, snr_choices=recipe.snr_choices,
        )
        train(data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
              n_per_class=0, seed=seed, model=model, dataset=data)

    curve = robustness_curve(
        model, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=seed + 1, threshold=threshold,
    )
    return RecipeResult(
        recipe=recipe,
        clean_acc=round(_clean_acc(curve), 4),
        reliable_floor_db=curve.reliable_floor_db,
        points=[p.to_dict() for p in curve.points],
    )


def augmentation_ab(
    recipes: list[AugRecipe],
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    threshold: float = 0.8,
    min_clean_acc: float = 0.9,
    out_dir: str | Path | None = None,
) -> AugmentationABResult:
    """Train each recipe and rank by reliable floor; recommend the best.

    Every recipe shares architecture, epochs, optimizer, and the *same* eval
    draw — only the training data differs, so the floor gap is the recipe's.
    The recommendation, among recipes that keep clean accuracy ≥
    ``min_clean_acc``, has the **deepest reliable floor**; ties break toward
    higher clean accuracy.
    """
    results = [
        _evaluate(
            r, snr_levels=snr_levels, n_per_class=n_per_class,
            eval_n_per_class=eval_n_per_class, epochs=epochs,
            batch_size=batch_size, lr=lr, seed=seed, threshold=threshold,
        )
        for r in recipes
    ]

    eligible = [r for r in results if r.clean_acc >= min_clean_acc]
    recommended = min(
        eligible,
        key=lambda r: (
            r.reliable_floor_db if r.reliable_floor_db is not None else _NO_FLOOR,
            -r.clean_acc,
        ),
        default=None,
    )

    out = AugmentationABResult(
        results=results, recommended=recommended, threshold=threshold,
        min_clean_acc=min_clean_acc,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "augmentation_ab.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "augmentation_ab.md").write_text(to_markdown(out))

    return out


def _snr_label(v: float | None) -> str:
    return "clean" if v is None else f"{v:g} dB"


def _floor_label(v: float | None) -> str:
    return "none (clean only)" if v is None else _snr_label(v)


def _recipe_label(r: AugRecipe) -> str:
    pool = ", ".join("clean" if s is None else f"{s:g}" for s in r.snr_choices)
    return f"{{{pool}}}"


def to_markdown(out: AugmentationABResult) -> str:
    rec = out.recommended
    if rec is None:
        verdict = (
            f"**No recipe keeps clean accuracy ≥ {out.min_clean_acc:.0%}** — "
            "every candidate trades away too much clean-audio accuracy. Loosen "
            "the augmentation or the clean-accuracy floor."
        )
    else:
        verdict = (
            f"**Recommended: `{rec.recipe.label}`** "
            f"(train SNRs {_recipe_label(rec.recipe)}) — reliable floor "
            f"**{_floor_label(rec.reliable_floor_db)}**, clean accuracy "
            f"{rec.clean_acc:.3f} (≥ {out.min_clean_acc:.0%})."
        )
    header = (
        "# Noise-augmentation recipe A/B\n\n"
        "Each recipe is a pool of training-time SNRs (per window: draw one, or "
        "stay clean). Same architecture, epochs, optimizer, and eval draw — only "
        "the training data differs. The winner has the **deepest reliable "
        f"floor** (lowest SNR still ≥ {out.threshold:.2f} accuracy) while keeping "
        f"clean accuracy ≥ {out.min_clean_acc:.0%}.\n\n"
        f"- {verdict}\n\n"
        "| recipe | train SNRs | clean acc | reliable floor |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for r in out.results:
        star = " ⭐" if r is rec else ""
        rows.append(
            f"| `{r.recipe.label}`{star} | {_recipe_label(r.recipe)} "
            f"| {r.clean_acc:.3f} | {_floor_label(r.reliable_floor_db)} |"
        )
    return header + "\n".join(rows) + "\n"


# Gentle → aggressive augmentation, to find where the floor stops improving.
_DEFAULT_RECIPES = [
    AugRecipe("clean-only", (None,)),
    AugRecipe("mild", (None, 20.0, 10.0)),
    AugRecipe("moderate", (None, 20.0, 10.0, 5.0)),
    AugRecipe("aggressive", (None, 20.0, 10.0, 5.0, 0.0, -5.0)),
]


def build_augmentation_ab(
    *,
    recipes: list[AugRecipe] | None = None,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    epochs: int = 12,
    seed: int = 0,
    out_dir: str | Path | None = "docs/benchmarks",
) -> AugmentationABResult:
    """Sweep the default gentle→aggressive recipes and write the artifact."""
    return augmentation_ab(
        recipes if recipes is not None else _DEFAULT_RECIPES,
        snr_levels=snr_levels, n_per_class=n_per_class,
        eval_n_per_class=eval_n_per_class, epochs=epochs, seed=seed,
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B the noise-augmentation recipe")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_augmentation_ab(
        epochs=args.epochs, n_per_class=args.n_per_class, seed=args.seed,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/augmentation_ab.json and augmentation_ab.md")


if __name__ == "__main__":
    main()
