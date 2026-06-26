"""fp32 -> INT8 confidence-calibration drift.

The shipped ``DEFAULT_TEMPERATURE = 0.41`` and the score-calibration analysis
(``model.calibration``) were both measured on the **fp32 eager** model. The device
serves the **INT8 ``.pte``** (``quantize_production``). PT2E quantization perturbs
activations, so the sigmoid scores the user actually sees on-device can carry a
different reliability profile than the one we calibrated — which would make the
calibrated-confidence read-out (and the fp32-fitted temperature) wrong on the
*deployed* model.

This module collects scores from the same noise-sweep eval through both the fp32
eager model and its INT8 ``.pte`` runtime (identical inputs), computes calibration
for each, and answers two questions:

1. **Calibration drift** — does INT8 change the raw ECE relative to fp32?
2. **Temperature transfer** — does the shipped fp32 temperature still reduce INT8
   ECE, and does it land near the temperature INT8 would pick for itself? If not,
   the deployed model needs its own re-fit.

Host-only; no device, no AI Hub token. The reduction is pure (unit-tested without
training); ``build_int8_calibration_drift`` trains one tiny net and quantizes it.

Run:
    PYTHONPATH=. .venv/bin/python -m model.int8_calibration_drift
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
    "CalibrationProfile",
    "Int8CalibrationDriftResult",
    "int8_calibration_drift",
    "build_int8_calibration_drift",
]

# How close two ECE values must be to count as "no meaningful drift". The
# score-calibration work treats ECE <= 0.05 as well-calibrated; half that is a
# conservative bar for "the quantizer didn't move the calibration".
_DRIFT_EPS = 0.025
# The shipped temperature "transfers" only if applying it to INT8 actually
# *reduces* the raw INT8 ECE (doesn't make it worse) by at least this margin.
_IMPROVE_EPS = 1e-4


@dataclass(frozen=True)
class CalibrationProfile:
    """Calibration of one model's scores: raw ECE + its own best temperature."""

    label: str
    n: int
    ece: float
    fitted_temperature: float
    ece_after_fitted: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Int8CalibrationDriftResult:
    """fp32 vs INT8 calibration, plus how the shipped temperature transfers."""

    fp32: CalibrationProfile
    int8: CalibrationProfile
    shipped_temperature: float
    int8_ece_shipped_temp: float
    n: int

    @property
    def ece_drift(self) -> float:
        """INT8 raw ECE minus fp32 raw ECE. Positive = INT8 less calibrated."""
        return self.int8.ece - self.fp32.ece

    @property
    def drift_is_small(self) -> bool:
        return abs(self.ece_drift) <= _DRIFT_EPS

    @property
    def shipped_temp_helps_int8(self) -> bool:
        """The fp32 temperature lowers (doesn't worsen) the raw INT8 ECE."""
        return self.int8_ece_shipped_temp <= self.int8.ece - _IMPROVE_EPS

    @property
    def temperature_transfers(self) -> bool:
        """The shipped fp32 temperature is safe to ship on the INT8 deployable.

        True when INT8 calibration didn't drift much AND applying the shipped
        temperature to INT8 still improves its ECE (so the constant we ship is
        the right *direction* and magnitude for the deployed model).
        """
        return self.drift_is_small and self.shipped_temp_helps_int8

    @property
    def verdict(self) -> str:
        drift = self.ece_drift
        direction = "raises" if drift > 0 else "lowers"
        base = (
            f"INT8 quantization {direction} ECE by {abs(drift):.3f} "
            f"(fp32 {self.fp32.ece:.3f} -> INT8 {self.int8.ece:.3f})"
        )
        if self.temperature_transfers:
            return (
                f"{base}; the drift is within {_DRIFT_EPS:g} and the shipped "
                f"fp32 temperature (T={self.shipped_temperature:g}) still cuts "
                f"INT8 ECE to {self.int8_ece_shipped_temp:.3f}, so it transfers "
                "to the deployed model — no re-fit needed."
            )
        if not self.drift_is_small:
            return (
                f"{base}; that exceeds the {_DRIFT_EPS:g} drift bar, so the INT8 "
                "deployable is calibrated differently from fp32 and should carry "
                "its own temperature "
                f"(INT8 re-fit T={self.int8.fitted_temperature:g})."
            )
        return (
            f"{base}; drift is small but the shipped fp32 temperature does not "
            f"improve INT8 ECE (stays {self.int8_ece_shipped_temp:.3f} vs raw "
            f"{self.int8.ece:.3f}) — re-fit on the INT8 model "
            f"(T={self.int8.fitted_temperature:g})."
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "shipped_temperature": self.shipped_temperature,
            "ece_drift": self.ece_drift,
            "drift_is_small": self.drift_is_small,
            "int8_ece_shipped_temp": self.int8_ece_shipped_temp,
            "shipped_temp_helps_int8": self.shipped_temp_helps_int8,
            "temperature_transfers": self.temperature_transfers,
            "verdict": self.verdict,
            "fp32": self.fp32.to_dict(),
            "int8": self.int8.to_dict(),
        }


def _profile(label: str, scores: torch.Tensor, labels: torch.Tensor, *, n_bins: int) -> CalibrationProfile:
    cal = compute_calibration(scores, labels, n_bins=n_bins, fit_temperature=True)
    return CalibrationProfile(
        label=label,
        n=int(scores.flatten().numel()),
        ece=float(cal.ece),
        fitted_temperature=float(cal.temperature),
        ece_after_fitted=float(cal.ece_after_temp),
    )


def int8_calibration_drift(
    fp32_scores: torch.Tensor,
    fp32_labels: torch.Tensor,
    int8_scores: torch.Tensor,
    int8_labels: torch.Tensor,
    *,
    shipped_temperature: float = DEFAULT_TEMPERATURE,
    n_bins: int = 10,
    out_dir: str | Path | None = None,
) -> Int8CalibrationDriftResult:
    """Compare fp32 vs INT8 calibration and the shipped temperature's transfer.

    All four inputs are flat tensors over the *same* eval; ``*_scores`` are
    sigmoid outputs in [0,1] and ``*_labels`` are 0/1. Pure reduction — no
    training, no device. Writes ``int8_calibration_drift.{json,md}`` to
    ``out_dir`` when given.
    """
    fp32_scores = fp32_scores.flatten().float()
    fp32_labels = fp32_labels.flatten().float()
    int8_scores = int8_scores.flatten().float()
    int8_labels = int8_labels.flatten().float()

    if fp32_scores.numel() == 0 or int8_scores.numel() == 0:
        raise ValueError("need at least one (score, label) pair for each model")
    if fp32_scores.numel() != fp32_labels.numel():
        raise ValueError("fp32 scores and labels differ in length")
    if int8_scores.numel() != int8_labels.numel():
        raise ValueError("int8 scores and labels differ in length")

    fp32 = _profile("fp32", fp32_scores, fp32_labels, n_bins=n_bins)
    int8 = _profile("int8", int8_scores, int8_labels, n_bins=n_bins)

    scaled = temperature_scale(int8_scores, shipped_temperature)
    int8_ece_shipped = compute_calibration(
        scaled, int8_labels, n_bins=n_bins, fit_temperature=False
    ).ece

    result = Int8CalibrationDriftResult(
        fp32=fp32,
        int8=int8,
        shipped_temperature=float(shipped_temperature),
        int8_ece_shipped_temp=float(int8_ece_shipped),
        n=int(fp32_scores.numel()),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "int8_calibration_drift.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        (out_dir / "int8_calibration_drift.md").write_text(to_markdown(result))

    return result


def to_markdown(out: Int8CalibrationDriftResult) -> str:
    d = out.to_dict()
    header = (
        "# fp32 -> INT8 confidence-calibration drift\n\n"
        "`DEFAULT_TEMPERATURE` and the score-calibration analysis were fit on the "
        "fp32 eager model, but the device serves the INT8 `.pte`. This checks "
        "whether quantization moves the calibration and whether the shipped fp32 "
        "temperature still fits the deployed model.\n\n"
        f"- windows (noise sweep): **{d['n']}**\n"
        f"- raw ECE: fp32 **{out.fp32.ece:.4f}** -> INT8 **{out.int8.ece:.4f}** "
        f"(drift **{d['ece_drift']:+.4f}**)\n"
        f"- own fitted temperature: fp32 **{out.fp32.fitted_temperature:g}** · "
        f"INT8 **{out.int8.fitted_temperature:g}**\n"
        f"- shipped T={out.shipped_temperature:g} applied to INT8 -> ECE "
        f"**{out.int8_ece_shipped_temp:.4f}** "
        f"(transfers: **{d['temperature_transfers']}**)\n"
        f"- **Verdict: {out.verdict}**\n\n"
        "| model | n | raw ECE | fitted T | ECE after fitted T |\n"
        "|---|---|---|---|---|\n"
        f"| fp32 (eager) | {out.fp32.n} | {out.fp32.ece:.4f} | "
        f"{out.fp32.fitted_temperature:g} | {out.fp32.ece_after_fitted:.4f} |\n"
        f"| INT8 (.pte) | {out.int8.n} | {out.int8.ece:.4f} | "
        f"{out.int8.fitted_temperature:g} | {out.int8.ece_after_fitted:.4f} |\n"
    )
    return header


def build_int8_calibration_drift(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 64,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    n_bins: int = 10,
    out_dir: str | Path | None = "docs/benchmarks",
) -> Int8CalibrationDriftResult:
    """Train one production net, quantize it, and measure fp32->INT8 drift.

    Imports inside the function to avoid a module-load cycle and to keep the
    heavy training stack off the import path of the pure reduction.
    """
    from .calibration import collect_scores
    from .int8_robustness import PteModule
    from .production import quantize_production, train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class,
        eval_n_per_class=eval_n_per_class, seed=seed, snr_levels=snr_levels,
    )
    pte = quantize_production(model)
    int8 = PteModule(pte)

    # Identical inputs to both models: same seed + same SNR grid.
    fp32_scores, fp32_labels = collect_scores(
        model, snr_levels=snr_levels, n_per_class=eval_n_per_class, seed=seed + 1
    )
    int8_scores, int8_labels = collect_scores(
        int8, snr_levels=snr_levels, n_per_class=eval_n_per_class, seed=seed + 1
    )

    return int8_calibration_drift(
        fp32_scores, fp32_labels, int8_scores, int8_labels,
        n_bins=n_bins, out_dir=out_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="fp32->INT8 calibration drift")
    p.add_argument("--out-dir", default="docs/benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n-per-class", type=int, default=96)
    args = p.parse_args()
    out = build_int8_calibration_drift(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/int8_calibration_drift.json and .md")


if __name__ == "__main__":
    main()
