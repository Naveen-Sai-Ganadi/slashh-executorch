"""A/B: global vs SNR-aware (oracle) temperature.

The per-SNR calibration breakdown (``model.calibration_snr``) showed the single
global ``DEFAULT_TEMPERATURE = 0.41`` under-corrects at the noisy floor — residual
ECE after temperature climbs from ~0.015 in the quiet to ~0.226 at -5 dB. This
A/B quantifies the *upper bound* on what an SNR-aware temperature could recover:
it pits the shipped global temperature against an oracle that knows each window's
true SNR and applies that SNR's own best temperature.

The oracle is not deployable (the device doesn't know the true SNR), so the gap
it recovers is an upper bound, not a shippable win — it's the prize an on-device
SNR estimator would chase. Small gap => the global temperature is good enough and
the estimator isn't worth building; large gap (especially at the floor) =>
SNR-aware calibration is on the table.

Host-only; no device, no AI Hub token. The reduction is pure (unit-tested
without training); ``build_snr_aware_temperature_ab`` trains one tiny net.

Run:
    PYTHONPATH=. .venv/bin/python -m model.snr_aware_temperature_ab
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .calibration import compute_calibration
from .confidence import DEFAULT_TEMPERATURE, temperature_scale

__all__ = [
    "SnrTempComparison",
    "SnrAwareTempResult",
    "snr_aware_temperature_ab",
    "build_snr_aware_temperature_ab",
]

# An SNR-aware temperature is "worth it" only if the oracle recovers at least
# this much ECE over the global temperature — pooled or at the floor. Below it,
# the extra machinery (an on-device SNR estimator) buys nothing meaningful.
_MATERIAL_ECE = 0.02

# The oracle fits per-SNR temperature on a fine grid that can descend BELOW the
# shipped DEFAULT_TEMPERATURE=0.41. That 0.41 is a regularization *floor* chosen
# in calibration_envelope to avoid the T->0 collapse on near-separable synthetic
# data: a temperature pinned to the grid floor here is degenerate (the same
# caveat the envelope reports), not a number to deploy. We let the oracle reach
# below 0.41 anyway so the A/B measures the *true* upper bound an SNR-aware
# schedule could chase, then flag when that bound is only reachable via
# degenerate sub-floor temperatures.
_ORACLE_LO = 0.05
_ORACLE_HI = 3.0
_ORACLE_STEP = 0.005


def _noise_rank(snr_db: float | None) -> float:
    return float("inf") if snr_db is None else snr_db


def _ece(scores: torch.Tensor, labels: torch.Tensor, *, n_bins: int) -> float:
    return float(
        compute_calibration(scores, labels, n_bins=n_bins, fit_temperature=False).ece
    )


def _fit_oracle_temperature(scores, labels, *, n_bins: int):
    """The temperature minimizing *this SNR's* ECE on a fine unconstrained grid.

    Searches ``[_ORACLE_LO, _ORACLE_HI]`` — which spans below the shipped 0.41
    floor — and minimizes the same score-calibration ECE the A/B reports, so the
    returned oracle ECE is the literal best-case for that SNR. ``floor_pinned`` is
    True when the optimum sits at the grid floor, marking a degenerate (not
    deployable) sub-0.41 temperature. The global 0.41 lies exactly on the grid,
    so the oracle can never do worse than the global temperature.
    """
    n_steps = int(round((_ORACLE_HI - _ORACLE_LO) / _ORACLE_STEP)) + 1
    best_t, best_ece = _ORACLE_LO, _ece(
        temperature_scale(scores, _ORACLE_LO), labels, n_bins=n_bins
    )
    for k in range(1, n_steps):
        t = _ORACLE_LO + _ORACLE_STEP * k
        e = _ece(temperature_scale(scores, t), labels, n_bins=n_bins)
        if e < best_ece:
            best_t, best_ece = t, e
    floor_pinned = best_t <= _ORACLE_LO + _ORACLE_STEP * 1.5
    return best_t, floor_pinned


@dataclass(frozen=True)
class SnrTempComparison:
    """Global vs oracle temperature at one SNR."""

    snr_db: float | None
    n: int
    ece_global: float
    ece_oracle: float
    oracle_temperature: float
    oracle_floor_pinned: bool = False

    @property
    def reduction(self) -> float:
        """ECE the oracle recovers over the global temperature (>=0)."""
        return self.ece_global - self.ece_oracle

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reduction"] = self.reduction
        return d


@dataclass(frozen=True)
class SnrAwareTempResult:
    """Per-SNR and pooled global-vs-oracle temperature comparison."""

    per_snr: list[SnrTempComparison]
    global_temperature: float
    pooled_ece_global: float
    pooled_ece_oracle: float

    @property
    def pooled_reduction(self) -> float:
        return self.pooled_ece_global - self.pooled_ece_oracle

    @property
    def floor(self) -> SnrTempComparison:
        return min(self.per_snr, key=lambda c: _noise_rank(c.snr_db))

    @property
    def material(self) -> bool:
        """Does the oracle recover a material ECE gap — pooled or at the floor?"""
        return (
            self.pooled_reduction >= _MATERIAL_ECE
            or self.floor.reduction >= _MATERIAL_ECE
        )

    @property
    def oracle_degenerate(self) -> bool:
        """Is the oracle's gain only reachable via degenerate sub-floor T?

        An oracle temperature pinned to the search floor is the same degeneracy
        ``calibration_envelope`` flags on near-separable synthetic audio: it is
        not a number to deploy. When the material gap exists only at SNRs whose
        oracle T is floor-pinned, the prize is real but unreachable safely.
        """
        return any(
            c.oracle_floor_pinned
            for c in self.per_snr
            if c.reduction >= _MATERIAL_ECE
        )

    @property
    def worth_it(self) -> bool:
        """Would a *deployable* SNR-aware schedule help?

        True only when the oracle recovers material ECE *without* relying on
        degenerate sub-floor temperatures. A material gap reachable only via
        floor-pinned T is not deployable on this evidence, so it does not count.
        """
        return self.material and not self.oracle_degenerate

    @property
    def verdict(self) -> str:
        f = self.floor
        floor_label = "clean" if f.snr_db is None else f"{f.snr_db:g} dB"
        if self.worth_it:
            return (
                f"an SNR-aware temperature is worth pursuing: an oracle that "
                f"knows the true SNR cuts pooled ECE by {self.pooled_reduction:.3f} "
                f"({self.pooled_ece_global:.3f}->{self.pooled_ece_oracle:.3f}) and "
                f"recovers {f.reduction:.3f} at the {floor_label} floor "
                f"({f.ece_global:.3f}->{f.ece_oracle:.3f}) — using temperatures "
                "inside the deployable range. That gap is the upper bound an "
                f"on-device SNR estimator could chase; the global "
                f"T={self.global_temperature:g} leaves it on the table."
            )
        if self.material and self.oracle_degenerate:
            return (
                f"the prize exists but isn't safely reachable: an oracle recovers "
                f"{f.reduction:.3f} ECE at the {floor_label} floor "
                f"({f.ece_global:.3f}->{f.ece_oracle:.3f}), but only by sharpening "
                f"below the {self.global_temperature:g} regularization floor — the "
                "same degenerate sub-floor temperature calibration_envelope flags "
                "on near-separable synthetic audio. An SNR-aware schedule can't "
                "claim this gain without real field audio to prove the sub-floor "
                "temperature generalizes; until then ship the single global "
                "temperature and treat low-SNR confidence as under-stated."
            )
        return (
            f"the global T={self.global_temperature:g} is good enough: an oracle "
            f"SNR-aware temperature recovers only {self.pooled_reduction:.3f} ECE "
            f"pooled and {f.reduction:.3f} at the {floor_label} floor — below the "
            f"{_MATERIAL_ECE:g} materiality bar. An on-device SNR estimator isn't "
            "worth building for calibration; ship the single global temperature."
        )

    def to_dict(self) -> dict:
        return {
            "global_temperature": self.global_temperature,
            "n_levels": len(self.per_snr),
            "pooled_ece_global": self.pooled_ece_global,
            "pooled_ece_oracle": self.pooled_ece_oracle,
            "pooled_reduction": self.pooled_reduction,
            "floor_snr_db": self.floor.snr_db,
            "floor_reduction": self.floor.reduction,
            "material": self.material,
            "oracle_degenerate": self.oracle_degenerate,
            "worth_it": self.worth_it,
            "verdict": self.verdict,
            "per_snr": [c.to_dict() for c in self.per_snr],
        }


def snr_aware_temperature_ab(
    triples,
    *,
    global_temperature: float = DEFAULT_TEMPERATURE,
    n_bins: int = 10,
    out_dir: str | Path | None = None,
) -> SnrAwareTempResult:
    """A/B the global temperature against an oracle per-SNR temperature.

    ``triples`` is an iterable of ``(snr_db, scores, labels)``; ``scores`` are
    sigmoid outputs in [0,1], ``labels`` are 0/1. Pure reduction — no training,
    no device. Writes ``snr_aware_temperature_ab.{json,md}`` to ``out_dir`` when
    given.
    """
    triples = [(snr, s.flatten().float(), y.flatten().float()) for snr, s, y in triples]
    if not triples:
        raise ValueError("need at least one (snr_db, scores, labels) tuple")

    per_snr: list[SnrTempComparison] = []
    global_pool_scores, oracle_pool_scores, pool_labels = [], [], []
    for snr, scores, labels in triples:
        # Oracle: this SNR's own ECE-minimizing temperature on a fine grid that
        # can descend below the 0.41 floor — the true per-SNR upper bound.
        oracle_t, floor_pinned = _fit_oracle_temperature(scores, labels, n_bins=n_bins)

        scaled_global = temperature_scale(scores, global_temperature)
        scaled_oracle = temperature_scale(scores, oracle_t)

        per_snr.append(
            SnrTempComparison(
                snr_db=snr,
                n=int(scores.numel()),
                ece_global=_ece(scaled_global, labels, n_bins=n_bins),
                ece_oracle=_ece(scaled_oracle, labels, n_bins=n_bins),
                oracle_temperature=oracle_t,
                oracle_floor_pinned=bool(floor_pinned),
            )
        )
        global_pool_scores.append(scaled_global)
        oracle_pool_scores.append(scaled_oracle)
        pool_labels.append(labels)

    pooled_labels = torch.cat(pool_labels)
    pooled_ece_global = _ece(torch.cat(global_pool_scores), pooled_labels, n_bins=n_bins)
    pooled_ece_oracle = _ece(torch.cat(oracle_pool_scores), pooled_labels, n_bins=n_bins)

    result = SnrAwareTempResult(
        per_snr=per_snr,
        global_temperature=float(global_temperature),
        pooled_ece_global=pooled_ece_global,
        pooled_ece_oracle=pooled_ece_oracle,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "snr_aware_temperature_ab.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        (out_dir / "snr_aware_temperature_ab.md").write_text(to_markdown(result))

    return result


def to_markdown(out: SnrAwareTempResult) -> str:
    d = out.to_dict()
    ordered = sorted(out.per_snr, key=lambda c: _noise_rank(c.snr_db), reverse=True)
    header = (
        "# SNR-aware vs global temperature (A/B)\n\n"
        "The per-SNR breakdown showed the single global temperature under-corrects "
        "at the noisy floor. This bounds the prize: an *oracle* that knows each "
        "window's true SNR applies that SNR's own best temperature. The oracle "
        "isn't deployable (the device doesn't know the true SNR) — the gap is the "
        "upper bound an on-device SNR estimator could chase.\n\n"
        f"- global `DEFAULT_TEMPERATURE`={out.global_temperature:g}; the oracle "
        f"fits each SNR's own T on a fine grid that may descend below 0.41 "
        "(sub-floor `(floor)` temperatures are degenerate — see "
        "calibration_envelope)\n"
        f"- pooled ECE: global **{out.pooled_ece_global:.4f}** -> oracle "
        f"**{out.pooled_ece_oracle:.4f}** (recovers **{d['pooled_reduction']:+.4f}**)\n"
        f"- floor reduction: **{d['floor_reduction']:+.4f}**  ·  oracle relies on "
        f"degenerate sub-floor T: **{d['oracle_degenerate']}**  ·  worth a "
        f"deployable SNR estimator: **{d['worth_it']}**\n"
        f"- **Verdict: {out.verdict}**\n\n"
        "| SNR | n | ECE (global T) | ECE (oracle T) | oracle T | recovered |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for c in ordered:
        label = "clean" if c.snr_db is None else f"{c.snr_db:g} dB"
        t = f"{c.oracle_temperature:g}{' (floor)' if c.oracle_floor_pinned else ''}"
        rows.append(
            f"| {label} | {c.n} | {c.ece_global:.4f} | {c.ece_oracle:.4f} | "
            f"{t} | {c.reduction:+.4f} |"
        )
    return header + "\n".join(rows) + "\n"


def build_snr_aware_temperature_ab(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    n_bins: int = 10,
    out_dir: str | Path | None = "docs/benchmarks",
) -> SnrAwareTempResult:
    """Train one production net and A/B global vs oracle SNR-aware temperature."""
    from .calibration import collect_scores
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class,
        eval_n_per_class=eval_n_per_class, seed=seed, snr_levels=snr_levels,
    )

    triples = []
    for snr in snr_levels:
        scores, labels = collect_scores(
            model, snr_levels=(snr,), n_per_class=eval_n_per_class, seed=seed + 1
        )
        triples.append((snr, scores, labels))

    return snr_aware_temperature_ab(triples, n_bins=n_bins, out_dir=out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="global vs SNR-aware temperature A/B")
    p.add_argument("--out-dir", default="docs/benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n-per-class", type=int, default=96)
    args = p.parse_args()
    out = build_snr_aware_temperature_ab(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/snr_aware_temperature_ab.json and .md")


if __name__ == "__main__":
    main()
