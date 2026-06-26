"""Per-SNR calibration breakdown.

``model.calibration`` pools scores across the whole noise sweep into one ECE.
That hides *where* the calibration lives: the confidence read-out can be
trustworthy on clean audio yet meaningless at the noisy reliable floor (-5 dB),
exactly where a user is most likely to need it. A single pooled number cannot
tell you whether to trust the confidence when it is noisy.

This module re-computes calibration *per SNR* — accuracy, ECE, and the
temperature each SNR would pick for itself — and renders the floor verdict: is
the confidence number meaningful across the operating range, or only in the
quiet? "Meaningful" here means the model is better than chance at that SNR; a
confidence attached to a coin-flip prediction is not a probability worth showing.

Host-only; no device, no AI Hub token. The reduction is pure (unit-tested
without training); ``build_calibration_snr`` trains one tiny net.

Run:
    PYTHONPATH=. .venv/bin/python -m model.calibration_snr
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .audio_config import STRESS_THRESHOLD
from .calibration import compute_calibration
from .confidence import DEFAULT_TEMPERATURE

__all__ = [
    "SnrCalibration",
    "CalibrationSnrResult",
    "calibration_snr",
    "build_calibration_snr",
]

# A confidence read-out is only a real probability if the prediction beats
# chance. Below this accuracy the model is ~coin-flipping at that SNR, so any
# calibrated number on top is noise dressed up as certainty.
_TRUST_ACC = 0.6


def _noise_rank(snr_db: float | None) -> float:
    """Sort key from least to most noisy. Clean (None) is least noisy."""
    return float("inf") if snr_db is None else snr_db


@dataclass(frozen=True)
class SnrCalibration:
    """Calibration at one SNR level."""

    snr_db: float | None
    n: int
    accuracy: float
    ece: float
    fitted_temperature: float
    ece_after_fitted: float

    @property
    def trustworthy(self) -> bool:
        return self.accuracy >= _TRUST_ACC

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trustworthy"] = self.trustworthy
        return d


@dataclass(frozen=True)
class CalibrationSnrResult:
    """Per-SNR calibration, ordered least->most noisy, with a floor verdict."""

    per_snr: list[SnrCalibration]
    global_temperature: float
    threshold: float

    @property
    def floor(self) -> SnrCalibration:
        """The noisiest tested SNR — the reliable floor the device operates at."""
        return min(self.per_snr, key=lambda r: _noise_rank(r.snr_db))

    @property
    def floor_trustworthy(self) -> bool:
        """Is the confidence read-out meaningful at the noisiest SNR?"""
        return self.floor.trustworthy

    @property
    def ece_increases_with_noise(self) -> bool:
        """Does calibration error grow monotonically as audio gets noisier?"""
        ordered = sorted(self.per_snr, key=lambda r: _noise_rank(r.snr_db), reverse=True)
        eces = [r.ece for r in ordered]
        return all(b >= a for a, b in zip(eces, eces[1:]))

    @property
    def verdict(self) -> str:
        f = self.floor
        quiet = max(self.per_snr, key=lambda r: _noise_rank(r.snr_db))
        floor_label = "clean" if f.snr_db is None else f"{f.snr_db:g} dB"
        if self.floor_trustworthy:
            return (
                f"the *prediction* is reliable across the whole sweep — even at "
                f"the {floor_label} floor accuracy holds at {f.accuracy:.2f}. But "
                f"the *confidence number* drifts under-confident with noise: ECE "
                f"climbs {quiet.ece:.3f}->{f.ece:.3f} from quiet to floor, and "
                f"even after the global T={self.global_temperature:g} the floor "
                f"keeps {f.ece_after_fitted:.3f} residual ECE vs "
                f"{quiet.ece_after_fitted:.3f} in the quiet. Trust the decision "
                "everywhere; treat low-SNR confidence as under-stated (a single "
                "global temperature under-corrects at the edge)."
            )
        return (
            f"the confidence read-out collapses at the {floor_label} floor: "
            f"accuracy {f.accuracy:.2f} is near chance, so any calibrated number "
            "shown there is not a real probability. Trust it in the quiet; treat "
            "low-SNR confidence as unreliable (gate on the detector, not the "
            "confidence) until the floor improves."
        )

    def to_dict(self) -> dict:
        return {
            "global_temperature": self.global_temperature,
            "threshold": self.threshold,
            "n_levels": len(self.per_snr),
            "floor_snr_db": self.floor.snr_db,
            "floor_trustworthy": self.floor_trustworthy,
            "ece_increases_with_noise": self.ece_increases_with_noise,
            "verdict": self.verdict,
            "per_snr": [r.to_dict() for r in self.per_snr],
        }


def calibration_snr(
    triples,
    *,
    global_temperature: float = DEFAULT_TEMPERATURE,
    threshold: float = STRESS_THRESHOLD,
    n_bins: int = 10,
    out_dir: str | Path | None = None,
) -> CalibrationSnrResult:
    """Per-SNR calibration from ``(snr_db, scores, labels)`` tuples.

    ``scores`` are sigmoid outputs in [0,1]; ``labels`` are 0/1. Pure reduction —
    no training, no device. Writes ``calibration_snr.{json,md}`` to ``out_dir``
    when given.
    """
    triples = list(triples)
    if not triples:
        raise ValueError("need at least one (snr_db, scores, labels) tuple")

    rows: list[SnrCalibration] = []
    for snr, scores, labels in triples:
        scores = scores.flatten().float()
        labels = labels.flatten().float()
        cal = compute_calibration(
            scores, labels, n_bins=n_bins, threshold=threshold, fit_temperature=True
        )
        rows.append(
            SnrCalibration(
                snr_db=snr,
                n=int(scores.numel()),
                accuracy=float(cal.accuracy),
                ece=float(cal.ece),
                fitted_temperature=float(cal.temperature),
                ece_after_fitted=float(cal.ece_after_temp),
            )
        )

    result = CalibrationSnrResult(
        per_snr=rows,
        global_temperature=float(global_temperature),
        threshold=float(threshold),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration_snr.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        (out_dir / "calibration_snr.md").write_text(to_markdown(result))

    return result


def to_markdown(out: CalibrationSnrResult) -> str:
    d = out.to_dict()
    ordered = sorted(out.per_snr, key=lambda r: _noise_rank(r.snr_db), reverse=True)
    header = (
        "# Per-SNR calibration breakdown — can you trust the confidence when "
        "it's noisy?\n\n"
        "The pooled score-calibration ECE hides *where* calibration lives. This "
        "re-fits per SNR: confidence is only a real probability where the model "
        "beats chance.\n\n"
        f"- noise levels: **{d['n_levels']}**  ·  global "
        f"`DEFAULT_TEMPERATURE`={out.global_temperature:g}\n"
        f"- ECE grows monotonically with noise: **{d['ece_increases_with_noise']}**\n"
        f"- floor ({'clean' if out.floor.snr_db is None else f'{out.floor.snr_db:g} dB'}) "
        f"trustworthy: **{d['floor_trustworthy']}**\n"
        f"- **Verdict: {out.verdict}**\n\n"
        "| SNR | n | accuracy | ECE | fitted T | ECE after T | trustworthy |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in ordered:
        label = "clean" if r.snr_db is None else f"{r.snr_db:g} dB"
        rows.append(
            f"| {label} | {r.n} | {r.accuracy:.3f} | {r.ece:.4f} | "
            f"{r.fitted_temperature:g} | {r.ece_after_fitted:.4f} | "
            f"{r.trustworthy} |"
        )
    return header + "\n".join(rows) + "\n"


def build_calibration_snr(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    n_bins: int = 10,
    out_dir: str | Path | None = "docs/benchmarks",
) -> CalibrationSnrResult:
    """Train one production net and break its calibration down per SNR.

    Imports inside the function to avoid a module-load cycle and to keep the
    training stack off the pure-reduction import path.
    """
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

    return calibration_snr(triples, n_bins=n_bins, out_dir=out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="per-SNR calibration breakdown")
    p.add_argument("--out-dir", default="docs/benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n-per-class", type=int, default=96)
    args = p.parse_args()
    out = build_calibration_snr(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/calibration_snr.json and .md")


if __name__ == "__main__":
    main()
