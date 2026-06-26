"""Recipe-parameterized cross-initialization envelope — does training on lower
SNRs deepen the *envelope*, or only a lucky seed's floor?

Two prior results meet here:

* ``init_envelope.py`` derives the **conservative envelope** — the worst-of-N
  per-init reliable floor, the SNR every init clears regardless of seed — but
  only for the shipped production recipe (pins it at 10 dB).
* ``augmentation_ab.py`` found, on a *single* seed, that a more aggressive
  training-noise recipe deepens the reliable floor (none -> 10 -> 0 -> -5 dB)
  at no clean-accuracy cost — and flagged that below 10 dB stays init-sensitive,
  so the envelope claim was "unchanged pending a multi-init confirmation."

This harness is that confirmation. For each recipe it trains several
independent inits, runs each through the same noise sweep, and reduces them to
a conservative envelope (reusing ``init_envelope``). Reporting the envelopes
side by side answers the honest question: a deeper *single-seed* floor only
moves the advertised number if the deeper *cross-init envelope* moves with it.

    python -m model.recipe_envelope --n-inits 4

Records ``docs/benchmarks/recipe_envelope.{json,md}``. Host-only; no device, no
AI Hub token.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .augmentation_ab import AugRecipe
from .init_envelope import InitFloor, init_envelope
from .model import StressNet
from .production import build_production_model
from .robust_train import noise_augmented_dataset
from .robustness import _snr_label
from .train import train

__all__ = [
    "RecipeEnvelope",
    "RecipeEnvelopeResult",
    "recipe_envelope",
    "build_recipe_envelope",
]


@dataclass(frozen=True)
class RecipeEnvelope:
    """One recipe's cross-init envelope and the per-init floors behind it."""

    recipe: AugRecipe
    envelope_db: float | None       # worst (least-noisy) per-init floor; None = none common
    mean_clean_acc: float           # mean clean accuracy across inits
    floors: list[dict]              # per-init InitFloor dicts (seed, clean_acc, floor)

    def to_dict(self) -> dict:
        return {
            "recipe": self.recipe.to_dict(),
            "envelope_db": self.envelope_db,
            "mean_clean_acc": self.mean_clean_acc,
            "floors": self.floors,
        }


@dataclass(frozen=True)
class RecipeEnvelopeResult:
    envelopes: list[RecipeEnvelope]
    deepest: RecipeEnvelope | None   # recipe whose envelope reaches the lowest SNR
    threshold: float
    n_inits: int

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_inits": self.n_inits,
            "deepest": self.deepest.to_dict() if self.deepest else None,
            "envelopes": [e.to_dict() for e in self.envelopes],
        }


# A recipe with no common noisy envelope must sort *worse* than any real floor.
_NO_ENVELOPE = float("inf")


def _train_recipe_inits(
    recipe: AugRecipe, *, n_inits: int, n_per_class: int, epochs: int,
    batch_size: int, lr: float, base_seed: int,
) -> list[tuple[int, StressNet]]:
    """Train ``n_inits`` independent inits of the shipped width on one recipe."""
    import torch

    models: list[tuple[int, StressNet]] = []
    for i in range(n_inits):
        seed = base_seed + i
        # Seed before constructing the net so weight init is reproducible and
        # independent of any prior global-RNG use (mirrors train_production).
        torch.manual_seed(seed)
        model = build_production_model()
        if recipe.snr_choices == (None,):
            train(data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
                  n_per_class=n_per_class, seed=seed, model=model)
        else:
            data = noise_augmented_dataset(
                n_per_class, seed=seed, snr_choices=recipe.snr_choices,
            )
            train(data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
                  n_per_class=0, seed=seed, model=model, dataset=data)
        models.append((seed, model))
    return models


def _envelope_for_recipe(
    recipe: AugRecipe, *, n_inits: int, snr_levels, n_per_class: int,
    eval_n_per_class: int, epochs: int, batch_size: int, lr: float,
    base_seed: int, eval_seed: int, threshold: float,
) -> RecipeEnvelope:
    models = _train_recipe_inits(
        recipe, n_inits=n_inits, n_per_class=n_per_class, epochs=epochs,
        batch_size=batch_size, lr=lr, base_seed=base_seed,
    )
    # Reuse the init-envelope reduction (same eval draw across inits).
    env = init_envelope(
        models, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        eval_seed=eval_seed, threshold=threshold, out_dir=None,
    )
    cleans = [f.clean_acc for f in env.floors]
    return RecipeEnvelope(
        recipe=recipe,
        envelope_db=env.envelope_db,
        mean_clean_acc=round(mean(cleans), 4) if cleans else float("nan"),
        floors=[f.to_dict() for f in env.floors],
    )


def recipe_envelope(
    recipes: Sequence[AugRecipe],
    *,
    n_inits: int = 4,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_per_class: int = 96,
    eval_n_per_class: int = 64,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    base_seed: int = 0,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> RecipeEnvelopeResult:
    """For each recipe, train ``n_inits`` inits and derive the cross-init envelope.

    Every recipe shares architecture, epochs, optimizer, init seeds, and the
    *same* eval draw — only the training-noise pool differs, so the envelope gap
    is the recipe's. ``deepest`` names the recipe whose conservative envelope
    reaches the lowest SNR (None envelopes — no common noisy floor — sort worst).
    """
    # All recipes share the same init seeds and eval draw so the comparison is
    # apples-to-apples: the eval seed sits well clear of the init-seed band.
    eval_seed = base_seed + 1000
    envelopes = [
        _envelope_for_recipe(
            r, n_inits=n_inits, snr_levels=snr_levels, n_per_class=n_per_class,
            eval_n_per_class=eval_n_per_class, epochs=epochs,
            batch_size=batch_size, lr=lr, base_seed=base_seed,
            eval_seed=eval_seed, threshold=threshold,
        )
        for r in recipes
    ]

    deepest = min(
        envelopes,
        key=lambda e: e.envelope_db if e.envelope_db is not None else _NO_ENVELOPE,
        default=None,
    )
    # If no recipe has a common noisy envelope, there is nothing to crown.
    if deepest is not None and deepest.envelope_db is None:
        deepest = None

    out = RecipeEnvelopeResult(
        envelopes=envelopes, deepest=deepest, threshold=threshold,
        n_inits=n_inits,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "recipe_envelope.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "recipe_envelope.md").write_text(to_markdown(out))

    return out


def _env_label(v: float | None) -> str:
    return "none (clean only)" if v is None else _snr_label(v)


def _recipe_pool(r: AugRecipe) -> str:
    return "{" + ", ".join("clean" if s is None else f"{s:g}" for s in r.snr_choices) + "}"


def to_markdown(out: RecipeEnvelopeResult) -> str:
    dee = out.deepest
    if dee is None:
        verdict = (
            "**No recipe yields a common noisy envelope** — for every recipe at "
            "least one init is reliable only on clean audio. The floor genuinely "
            "depends on the draw; advertise robustness on clean audio."
        )
    else:
        # Is the deepest envelope actually deeper than the others, or a tie?
        others = [e for e in out.envelopes if e is not dee]
        baseline = max(
            (e.envelope_db for e in others if e.envelope_db is not None),
            default=None,
        )
        if baseline is None or dee.envelope_db is None or dee.envelope_db >= baseline:
            moved = (
                "this matches the other recipes' envelope — the deeper "
                "single-seed floor does **not** move the cross-init envelope, so "
                "the advertised floor is unchanged (the single-seed gain was "
                "init-sensitive, as flagged)."
            )
        else:
            moved = (
                f"this is **deeper** than the next-best envelope "
                f"({_env_label(baseline)}) — the recipe moves the cross-init "
                "envelope, not just one seed's floor, so the advertised floor "
                "can follow it down."
            )
        verdict = (
            f"**Deepest cross-init envelope: `{dee.recipe.label}` "
            f"(train SNRs {_recipe_pool(dee.recipe)}) reliable down to "
            f"{_env_label(dee.envelope_db)}** across {out.n_inits} inits — {moved}"
        )
    header = (
        "# Recipe-parameterized cross-initialization envelope\n\n"
        f"For each recipe, {out.n_inits} independent inits of the shipped width "
        "were trained and run through the same noise sweep (same eval draw). The "
        "**envelope** is the least-noisy per-init floor — the SNR every init "
        "still clears regardless of the training seed. A deeper *single-seed* "
        "floor only moves the advertised number if the *envelope* moves with "
        f"it (threshold {out.threshold:.2f}).\n\n"
        f"- {verdict}\n\n"
        "| recipe | train SNRs | mean clean acc | cross-init envelope |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for e in out.envelopes:
        star = " ⭐" if e is dee else ""
        rows.append(
            f"| `{e.recipe.label}`{star} | {_recipe_pool(e.recipe)} "
            f"| {e.mean_clean_acc:.3f} | {_env_label(e.envelope_db)} |"
        )
    return header + "\n".join(rows) + "\n"


# The promotion ladder: the previous production recipe (`aggressive`, envelope
# 0 dB) vs the shipped production recipe (`very-aggressive`, adds a -10 dB window)
# vs going one notch deeper (`extreme`, adds -15 dB). This three-way settles both
# the promotion (does -10 dB move the envelope past 0 dB?) and the stopping point
# (does -15 dB move it further, or has the augmentation approach hit its floor?).
_DEFAULT_RECIPES = [
    AugRecipe("aggressive", (None, 20.0, 10.0, 5.0, 0.0, -5.0)),
    AugRecipe("very-aggressive", (None, 20.0, 10.0, 5.0, 0.0, -5.0, -10.0)),
    AugRecipe("extreme", (None, 20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0)),
]


def build_recipe_envelope(
    *,
    recipes: Sequence[AugRecipe] | None = None,
    n_inits: int = 4,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0, -15.0),
    n_per_class: int = 96,
    eval_n_per_class: int = 64,
    epochs: int = 12,
    base_seed: int = 0,
    out_dir: str | Path | None = "docs/benchmarks",
) -> RecipeEnvelopeResult:
    """Run the aggressive→very-aggressive→extreme ladder and write the artifact."""
    return recipe_envelope(
        recipes if recipes is not None else _DEFAULT_RECIPES,
        n_inits=n_inits, snr_levels=snr_levels, n_per_class=n_per_class,
        eval_n_per_class=eval_n_per_class, epochs=epochs, base_seed=base_seed,
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Recipe-parameterized cross-init envelope")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--n-inits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    out = build_recipe_envelope(
        n_inits=args.n_inits, epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, base_seed=args.base_seed,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/recipe_envelope.json and recipe_envelope.md")


if __name__ == "__main__":
    main()
