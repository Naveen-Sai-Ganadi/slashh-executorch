"""Cross-initialization calibration-temperature envelope.

``model/confidence.py`` ships ``DEFAULT_TEMPERATURE = 0.41`` — the temperature
that recalibrates the production net's under-confident scores. That constant was
fit by ``model.calibration`` on a *single* initialization (seed 0). But a
1,549-param net is init-sensitive: this project advertises a noise-robustness
*floor* only after measuring it across several inits (``init_envelope`` /
``recipe_envelope``) precisely because a single draw can mislead. A calibration
temperature deserves the same scrutiny.

This harness re-fits the temperature across N independent inits of the
production recipe and reports the spread, then answers one question: is the
shipped ``DEFAULT_TEMPERATURE`` a *stable* default — bracketed by the per-init
fits and pointing the same direction (all under-confident) — or seed-luck? Either
answer is shippable evidence: it tells us whether 0.41 is a number to stand on or
a constant that needs widening guards.

    python -m model.calibration_envelope        # train N inits, fit T each, record

Records ``docs/benchmarks/calibration_envelope.{json,md}``. Host-only; the only
compute is host training + temperature fitting. No device, no AI Hub token.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .confidence import DEFAULT_TEMPERATURE

__all__ = [
    "CalibrationFit",
    "CalibrationEnvelopeResult",
    "calibration_envelope",
    "build_calibration_envelope",
]

# A spread this tight (max−min) counts as "tight" for the stable-default verdict.
_TIGHT_SPREAD = 0.15


@dataclass(frozen=True)
class CalibrationFit:
    """One initialization's fitted temperatures and its before/after ECE.

    ``temperature`` is the *deployable* estimate — the regularized fit produced by
    the same routine that chose the shipped 0.41 (its grid floors at 0.41 to avoid
    the T→0 collapse on near-separable data). ``unconstrained_temperature`` is the
    free NLL optimum on a fine grid that can descend below 0.41; when it sits at
    the grid floor (``unconstrained_floor_pinned``) the optimum is degenerate —
    a signal that the eval set cannot identify a finite temperature.
    """

    seed: int
    temperature: float
    ece: float
    ece_after_temp: float
    unconstrained_temperature: float | None = None
    unconstrained_floor_pinned: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationEnvelopeResult:
    """The per-init temperature fits reduced to a spread + stability verdict."""

    fits: list[CalibrationFit]
    default_temperature: float

    @property
    def temperatures(self) -> list[float]:
        return [f.temperature for f in self.fits]

    @property
    def all_same_direction(self) -> bool:
        """True if every init's fit points the same way relative to T=1.

        T<1 sharpens (corrects under-confidence); T>1 softens (corrects
        over-confidence). If all inits agree on the direction, recalibration has a
        consistent meaning; a mix means the direction itself is a draw artifact.
        """
        temps = self.temperatures
        return all(t < 1.0 for t in temps) or all(t > 1.0 for t in temps)

    @property
    def default_within_spread(self) -> bool:
        """True if the shipped default is bracketed by the observed per-init fits."""
        temps = self.temperatures
        return min(temps) <= self.default_temperature <= max(temps)

    @property
    def is_stable(self) -> bool:
        """A stable default: tight spread, one direction, and bracketed."""
        temps = self.temperatures
        spread = max(temps) - min(temps)
        return self.all_same_direction and self.default_within_spread and spread <= _TIGHT_SPREAD

    @property
    def unconstrained_degenerate(self) -> bool:
        """True if any init's *free* NLL optimum collapsed to the grid floor.

        A floor-pinned unconstrained optimum means the eval set is near-separable
        in score-ranking, so the NLL keeps falling as T→0 and the temperature is
        not finitely identifiable from this data — the deployable 0.41 is a
        regularized choice, not a measured optimum.
        """
        return any(f.unconstrained_floor_pinned for f in self.fits)

    @property
    def verdict(self) -> str:
        if self.is_stable:
            base = (
                "stable default — every initialization's deployable (regularized) "
                "temperature lands on the same value and the shipped "
                "DEFAULT_TEMPERATURE sits inside that spread, so 0.41 is "
                "init-independent, not seed-luck"
            )
        elif not self.all_same_direction:
            base = (
                "direction unstable — inits disagree on whether the model is over- "
                "or under-confident; a single fixed temperature is not justified"
            )
        elif not self.default_within_spread:
            base = (
                "default off-center — the shipped DEFAULT_TEMPERATURE lies outside "
                "the per-init spread; consider re-fitting it on the median"
            )
        else:
            base = (
                "wide but consistent — the fits agree in direction and bracket the "
                "default, but the spread is large; treat 0.41 as approximate"
            )
        if self.unconstrained_degenerate:
            base += (
                ". Caveat: the unconstrained NLL optimum is degenerate (collapses "
                "to the search floor) on the near-separable synthetic eval, so the "
                "temperature is not finitely identifiable here — re-fit on labeled "
                "field audio before trusting any value below 0.41"
            )
        return base

    def _unconstrained_block(self) -> dict | None:
        vals = [
            f.unconstrained_temperature
            for f in self.fits
            if f.unconstrained_temperature is not None
        ]
        if not vals:
            return None
        return {
            "temp_min": min(vals),
            "temp_median": statistics.median(vals),
            "temp_max": max(vals),
            "floor_pinned_count": sum(1 for f in self.fits if f.unconstrained_floor_pinned),
            "degenerate": self.unconstrained_degenerate,
        }

    def to_dict(self) -> dict:
        temps = self.temperatures
        d = {
            "default_temperature": self.default_temperature,
            "n_inits": len(self.fits),
            "temp_min": min(temps),
            "temp_median": statistics.median(temps),
            "temp_max": max(temps),
            "temp_mean": statistics.fmean(temps),
            "temp_std": statistics.pstdev(temps) if len(temps) > 1 else 0.0,
            "all_same_direction": self.all_same_direction,
            "default_within_spread": self.default_within_spread,
            "is_stable": self.is_stable,
            "verdict": self.verdict,
            "fits": [f.to_dict() for f in self.fits],
        }
        unconstrained = self._unconstrained_block()
        if unconstrained is not None:
            d["unconstrained"] = unconstrained
        return d


def calibration_envelope(
    fits: Sequence[CalibrationFit],
    *,
    default_temperature: float = DEFAULT_TEMPERATURE,
    out_dir: str | Path | None = None,
) -> CalibrationEnvelopeResult:
    """Reduce per-init temperature fits to a spread + stability verdict.

    ``fits`` is one :class:`CalibrationFit` per initialization. Pure reduction —
    no training — so the verdict logic is testable in isolation.
    """
    if not fits:
        raise ValueError("need at least one CalibrationFit")

    out = CalibrationEnvelopeResult(
        fits=list(fits), default_temperature=default_temperature
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration_envelope.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "calibration_envelope.md").write_text(to_markdown(out))

    return out


def to_markdown(out: CalibrationEnvelopeResult) -> str:
    d = out.to_dict()
    header = (
        "# Cross-initialization calibration-temperature envelope\n\n"
        f"{d['n_inits']} independent initializations of the production recipe were "
        "trained, and the post-hoc temperature was re-fit on each (same eval draw "
        "per init). The shipped `DEFAULT_TEMPERATURE` is validated against the "
        "resulting spread.\n\n"
        f"- shipped `DEFAULT_TEMPERATURE`: **{out.default_temperature:g}**\n"
        f"- per-init deployable temperature: **{d['temp_min']:.3f} … {d['temp_max']:.3f}** "
        f"(median {d['temp_median']:.3f}, mean {d['temp_mean']:.3f} ± "
        f"{d['temp_std']:.3f})\n"
    )
    unc = d.get("unconstrained")
    if unc is not None:
        header += (
            f"- per-init *unconstrained* optimum: **{unc['temp_min']:.3f} … "
            f"{unc['temp_max']:.3f}** "
            f"({unc['floor_pinned_count']}/{d['n_inits']} pinned to the search floor"
            f"{' — **degenerate**' if unc['degenerate'] else ''})\n"
        )
    header += (
        f"- **Verdict: {out.verdict}.**\n\n"
        "| init (seed) | deployable T | unconstrained T | ECE | ECE after T |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for f in out.fits:
        unc_t = (
            f"{f.unconstrained_temperature:.3f}"
            f"{' (floor)' if f.unconstrained_floor_pinned else ''}"
            if f.unconstrained_temperature is not None
            else "—"
        )
        rows.append(
            f"| {f.seed} | {f.temperature:.3f} | {unc_t} | {f.ece:.3f} | "
            f"{f.ece_after_temp:.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def _fit_temperature_unconstrained(
    scores, labels, *, n_bins: int, lo: float = 0.05, hi: float = 3.0, step: float = 0.005
):
    """Fine 1-D grid fit of the NLL-optimal temperature over ``[lo, hi]``.

    Why not reuse ``calibration._fit_temperature``? That routine's coarse grid
    starts at 0.5 and only refines ±0.09 below the best coarse point, so the
    lowest temperature it can ever return is **0.41** — which is exactly the
    shipped ``DEFAULT_TEMPERATURE``. An under-confident model whose true optimum
    is below 0.41 is silently clamped there, hiding any per-init spread. To
    measure whether 0.41 is a free optimum or a search floor, we fit the
    *unconstrained* optimum on a fine grid that can descend below it.
    """
    import torch

    from .calibration import _binned, _confidence_and_correct, _ece_mce

    eps = 1e-6
    p = scores.clamp(eps, 1.0 - eps)
    logits = torch.log(p / (1.0 - p))
    y = labels

    def nll(T: float) -> float:
        q = torch.sigmoid(logits / T).clamp(eps, 1.0 - eps)
        return float(-(y * torch.log(q) + (1 - y) * torch.log(1 - q)).mean())

    n_steps = int(round((hi - lo) / step)) + 1
    best_T, best_nll = lo, nll(lo)
    for k in range(1, n_steps):
        T = lo + step * k
        v = nll(T)
        if v < best_nll:
            best_T, best_nll = T, v

    floor_pinned = best_T <= lo + step * 1.5
    q = torch.sigmoid(logits / best_T)
    conf, correct = _confidence_and_correct(q, y)
    ece_after, _, _ = _ece_mce(_binned(conf, correct, n_bins), int(y.numel()))
    return best_T, ece_after, floor_pinned


def build_calibration_envelope(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    snr_levels: Sequence[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    n_bins: int = 10,
    out_dir: str | Path | None = "docs/benchmarks",
) -> CalibrationEnvelopeResult:
    """Train one production net per seed, re-fit the *unconstrained* temperature.

    Each seed gets an independent init (``train_production`` seeds the global RNG
    before constructing the net) and is scored over the same noise sweep. Unlike
    the shipped routine — whose grid floors at 0.41 — we fit the unconstrained
    optimum (see :func:`_fit_temperature_unconstrained`), so the envelope reveals
    where the per-init optima actually land relative to the shipped 0.41.
    """
    # Imported here to avoid a module-load cycle (calibration imports production,
    # production imports robustness).
    from .calibration import collect_scores, compute_calibration
    from .production import train_production

    fits: list[CalibrationFit] = []
    for seed in seeds:
        model, _ = train_production(
            epochs=epochs, n_per_class=n_per_class, seed=seed, snr_levels=list(snr_levels),
        )
        scores, labels = collect_scores(
            model, snr_levels=snr_levels, n_per_class=eval_n_per_class, seed=seed + 1,
        )
        cal = compute_calibration(scores, labels, n_bins=n_bins, fit_temperature=True)
        t_free, _, floor_pinned = _fit_temperature_unconstrained(scores, labels, n_bins=n_bins)
        fits.append(
            CalibrationFit(
                seed=seed,
                temperature=round(cal.temperature, 4),
                ece=round(cal.ece, 4),
                ece_after_temp=round(cal.ece_after_temp, 4),
                unconstrained_temperature=round(float(t_free), 4),
                unconstrained_floor_pinned=bool(floor_pinned),
            )
        )

    return calibration_envelope(fits, out_dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-init calibration-temperature envelope"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()

    out = build_calibration_envelope(
        seeds=args.seeds, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(
        f"wrote {args.out_dir}/calibration_envelope.json and calibration_envelope.md"
    )


if __name__ == "__main__":
    main()
