"""Base-rate (prior-shift) precision A/B.

Every accuracy / F1 / robustness artifact in this repo evaluates on a *balanced*
set — 50% stressed, 50% calm. Real usage is nothing like that: a phone listening
through a day hears overwhelmingly calm audio. Under that skew even a small
per-window false-alarm rate dominates the alarms the user actually experiences,
because calm windows vastly outnumber stressed ones — the base-rate fallacy. A
detector that looks great at 50/50 ("96% accurate!") can still cry wolf most of
the time in the field.

This A/B closes that gap. It measures the model's per-window TPR (sensitivity)
and FPR (1 − specificity) at the operating threshold, then applies Bayes to
report the realized **precision of an alarm** — P(stressed | alarm) — across a
sweep of stress prevalences, plus the *break-even prevalence* where precision
crosses 0.5 (below it, most alarms are false). That break-even number, compared
against a realistic deployment prevalence, is the honest "is this usable in the
wild" answer that balanced accuracy hides.

Bayes, per SNR:
    precision(p) = tpr·p / (tpr·p + fpr·(1−p))
    break-even  = fpr / (tpr + fpr)        # prevalence where precision = 0.5

Host-only: CPU training + thresholding, no device, no AI Hub token. The
reduction is pure (unit-tested with hand-set TPR/FPR); ``build_base_rate_precision``
trains one tiny net and measures TPR/FPR per SNR.

Run:
    PYTHONPATH=. .venv/bin/python -m model.base_rate_precision
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BaseRatePoint",
    "BaseRateResult",
    "base_rate_precision",
    "build_base_rate_precision",
]

# Precision below this at the realistic prevalence means "most alarms are false".
_DEFAULT_PRECISION_BAR = 0.5


@dataclass(frozen=True)
class BaseRatePoint:
    """Per-window detector operating point at one SNR (``None`` = clean).

    ``tpr`` = P(alarm | stressed); ``fpr`` = P(alarm | calm). Both are measured
    on a balanced eval set, then re-weighted by an arbitrary prior via Bayes.
    """

    snr_db: float | None
    tpr: float
    fpr: float
    n: int  # windows per class behind tpr/fpr at this SNR

    def precision_at(self, prior: float) -> float | None:
        """Realized P(stressed | alarm) at stress prevalence ``prior``.

        ``None`` when the model never alarms (tpr = fpr = 0): precision is
        undefined, not zero — there are simply no alarms to be right or wrong.
        """
        denom = self.tpr * prior + self.fpr * (1.0 - prior)
        if denom <= 0.0:
            return None
        return self.tpr * prior / denom

    @property
    def break_even_prior(self) -> float | None:
        """Stress prevalence at which precision crosses 0.5.

        Below it most alarms are false; above it most are real. ``fpr = 0`` →
        precision is 1.0 at any positive prevalence, so break-even is 0.0.
        ``tpr = fpr = 0`` (never alarms) → ``None``.
        """
        denom = self.tpr + self.fpr
        if denom <= 0.0:
            return None
        return self.fpr / denom

    def to_dict(self, priors) -> dict:
        return {
            "snr_db": self.snr_db,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "n": self.n,
            "break_even_prior": self.break_even_prior,
            "precision_by_prior": {
                repr(p): self.precision_at(p) for p in priors
            },
        }


@dataclass(frozen=True)
class BaseRateResult:
    """Base-rate precision A/B across an SNR sweep at a fixed set of priors."""

    points: list[BaseRatePoint]
    priors: tuple[float, ...]
    precision_bar: float

    @property
    def realistic_prior(self) -> float:
        """The most demanding (smallest) swept prevalence — the field condition."""
        return min(self.priors)

    @property
    def worst_break_even(self) -> float | None:
        """Highest break-even prevalence across SNRs (the hardest SNR to trust).

        ``None`` only if no SNR ever alarms. SNRs that never alarm are skipped.
        """
        bes = [p.break_even_prior for p in self.points if p.break_even_prior is not None]
        return max(bes) if bes else None

    @property
    def worst_precision_at_realistic(self) -> float | None:
        """Lowest alarm precision at the realistic prevalence across SNRs."""
        ps = [
            p.precision_at(self.realistic_prior)
            for p in self.points
            if p.precision_at(self.realistic_prior) is not None
        ]
        return min(ps) if ps else None

    @property
    def usable_at_realistic(self) -> bool:
        """True if every SNR clears the precision bar at the realistic prevalence.

        A model that never alarms at some SNR is not "usable" there — it is
        silent, so it fails the bar.
        """
        for p in self.points:
            prec = p.precision_at(self.realistic_prior)
            if prec is None or prec < self.precision_bar:
                return False
        return True

    @property
    def verdict(self) -> str:
        rp = self.realistic_prior
        wbe = self.worst_break_even
        if self.usable_at_realistic:
            wp = self.worst_precision_at_realistic
            return (
                f"**Usable under field skew**: at a realistic {rp:.1%} stress "
                f"prevalence every SNR keeps alarm precision ≥ "
                f"{self.precision_bar:.0%} (worst {wp:.1%}). Break-even prevalence "
                f"tops out at {wbe:.2%}, comfortably below deployment reality — "
                "most alarms the user sees are real."
            )
        if wbe is None:
            return (
                "**Detector is silent**: at the operating threshold the model "
                "never alarms at the measured SNRs, so precision is undefined. "
                "The threshold is too high to be useful in the field."
            )
        worst = max(
            (p for p in self.points if p.break_even_prior is not None),
            key=lambda p: p.break_even_prior,
        )
        return (
            f"**Base-rate limited**: alarm precision falls below "
            f"{self.precision_bar:.0%} at the realistic {rp:.1%} prevalence "
            f"(break-even needs {wbe:.2%}, worst at {_snr_label(worst.snr_db)}). "
            "Under real calm-dominated audio most alarms would be false — the "
            "per-window false-alarm rate, not balanced accuracy, is the ceiling. "
            "Lift specificity (raise the threshold or add temporal gating) before "
            "trusting alarms in the wild."
        )

    def to_dict(self) -> dict:
        return {
            "priors": list(self.priors),
            "precision_bar": self.precision_bar,
            "realistic_prior": self.realistic_prior,
            "worst_break_even": self.worst_break_even,
            "worst_precision_at_realistic": self.worst_precision_at_realistic,
            "usable_at_realistic": self.usable_at_realistic,
            "verdict": self.verdict,
            "points": [p.to_dict(self.priors) for p in self.points],
        }


def base_rate_precision(
    records,
    *,
    priors,
    precision_bar: float = _DEFAULT_PRECISION_BAR,
    out_dir: str | Path | None = None,
) -> BaseRateResult:
    """Reduce per-SNR ``(snr_db, tpr, fpr, n)`` operating points to a verdict.

    ``priors`` is a sweep of stress prevalences in the open interval (0, 1) —
    e.g. ``(0.5, 0.1, 0.02)`` from balanced down to a realistic field rate. Pure:
    no training, no device. Writes ``base_rate_precision.{json,md}`` to
    ``out_dir`` when given.
    """
    priors = tuple(float(p) for p in priors)
    if not priors:
        raise ValueError("priors must be non-empty")
    if any(not (0.0 < p < 1.0) for p in priors):
        raise ValueError("each prior must be in the open interval (0, 1)")

    points = [
        BaseRatePoint(snr_db=s, tpr=float(t), fpr=float(f), n=int(n))
        for s, t, f, n in records
    ]
    if not points:
        raise ValueError("records must be non-empty")

    out = BaseRateResult(points=points, priors=priors, precision_bar=precision_bar)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "base_rate_precision.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "base_rate_precision.md").write_text(to_markdown(out))

    return out


def _snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:.2%}" if pct else f"{x:.4f}"


def to_markdown(out: BaseRateResult) -> str:
    rp = out.realistic_prior
    header = (
        "# Base-rate (prior-shift) precision (A/B)\n\n"
        "Balanced accuracy assumes stress is 50% of windows. In the field a phone "
        "hears overwhelmingly calm audio, so even a small per-window false-alarm "
        "rate makes most alarms false (the base-rate fallacy). Each row takes the "
        "detector's measured **TPR/FPR** at the operating threshold and applies "
        "Bayes to report the realized **precision of an alarm** at several stress "
        "prevalences, plus the break-even prevalence where precision crosses "
        f"{out.precision_bar:.0%}.\n\n"
        f"- realistic prevalence swept: **{rp:.1%}**\n"
        f"- worst break-even prevalence: **{_fmt(out.worst_break_even, pct=True)}**\n"
        f"- worst alarm precision @ {rp:.1%}: "
        f"**{_fmt(out.worst_precision_at_realistic, pct=True)}**\n"
        f"- usable at realistic prevalence: **{out.usable_at_realistic}**\n"
        f"- {out.verdict}\n\n"
    )
    prior_cols = " | ".join(f"P@{p:g}" for p in out.priors)
    table_head = (
        f"| SNR | TPR | FPR | break-even | {prior_cols} |\n"
        + "|---|---|---|---|" + "---|" * len(out.priors) + "\n"
    )
    rows = []
    for p in out.points:
        precs = " | ".join(_fmt(p.precision_at(pr), pct=True) for pr in out.priors)
        rows.append(
            f"| {_snr_label(p.snr_db)} | {p.tpr:.3f} | {p.fpr:.3f} | "
            f"{_fmt(p.break_even_prior, pct=True)} | {precs} |"
        )
    return header + table_head + "\n".join(rows) + "\n"


def build_base_rate_precision(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    priors=(0.5, 0.1, 0.02),
    precision_bar: float = _DEFAULT_PRECISION_BAR,
    out_dir: str | Path | None = "docs/benchmarks",
) -> BaseRateResult:
    """Train one net, measure per-SNR TPR/FPR at the operating threshold.

    The model is the shipped fp32 production net (trained noise-augmented at the
    internal 0.8 head threshold). Evaluation thresholds scores at the *deployed*
    ``STRESS_THRESHOLD`` (0.6) — the threshold the on-device detector enters
    "stressed" at — so TPR/FPR reflect the real operating point. Imports inside
    the function to keep torch/training off the pure reduction path.
    """
    import torch

    from .audio_config import STRESS_THRESHOLD
    from .production import train_production
    from .robustness import noisy_synthetic_dataset

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=0.8,
    )
    model = model.eval()

    records = []
    for snr in snr_levels:
        x, y = noisy_synthetic_dataset(eval_n_per_class, seed + 7, snr_db=snr)
        with torch.no_grad():
            scores = model(x).flatten()
        alarm = scores >= STRESS_THRESHOLD
        y = y.flatten().bool()
        pos = int(y.sum())
        neg = int((~y).sum())
        tpr = float((alarm & y).sum()) / pos if pos else 0.0
        fpr = float((alarm & ~y).sum()) / neg if neg else 0.0
        records.append((snr, tpr, fpr, min(pos, neg)))

    return base_rate_precision(
        records, priors=priors, precision_bar=precision_bar, out_dir=out_dir
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Base-rate (prior-shift) precision A/B"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_base_rate_precision(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/base_rate_precision.json and .md")


if __name__ == "__main__":
    main()
