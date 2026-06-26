"""Which *way* does StressNet fail as noise rises?

``model/robustness.py`` reports aggregate accuracy vs SNR — necessary, but it
hides the asymmetry that matters for a stress detector. Two failures with the
same accuracy are not the same product:

* **misses_stress** — false negatives dominate; the model predicts "calm" on
  stressed audio and the detector goes *silent* when it should fire.
* **false_alarms** — false positives dominate; the model fires on calm audio and
  the detector *cries wolf*.

They call for opposite mitigations at the detector gate (model/detector.py): a
silent model wants a *lower* stress threshold / faster EMA attack; a trigger-
happy one wants a *higher* threshold / more smoothing. This harness re-runs the
same waveform-noise SNR sweep, splits each point into precision/recall + the
dominant error direction, and names the failure at the first SNR that drops
below the reliability threshold.

    python -m model.noise_failure_mode        # train, sweep, record the breakdown

Records ``docs/benchmarks/noise_failure_mode.{json,md}``. Host-only; no device,
no AI Hub token.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .eval import _confusion
from .model import StressNet
from .robustness import _snr_label, noisy_synthetic_dataset

__all__ = [
    "DIRECTIONS",
    "FailureModePoint",
    "FailureModeResult",
    "noise_failure_mode",
    "build_noise_failure_mode",
]

# Error-direction labels. "balanced" = neither false negatives nor false
# positives clearly dominate (incl. the no-error case).
DIRECTIONS = ("balanced", "misses_stress", "false_alarms")

# A direction is only called when one error type is at least this many times the
# other; below the margin the errors are too close to label honestly.
_MARGIN = 2.0


def _direction(fp: int, fn: int) -> str:
    """Name the dominant error direction from false-positive/negative counts."""
    if fp == 0 and fn == 0:
        return "balanced"
    if fn >= _MARGIN * max(fp, 1):
        return "misses_stress"
    if fp >= _MARGIN * max(fn, 1):
        return "false_alarms"
    return "balanced"


@dataclass(frozen=True)
class FailureModePoint:
    """Per-SNR error breakdown (``snr_db=None`` = clean)."""

    snr_db: float | None
    accuracy: float
    precision: float
    recall: float
    tp: int
    tn: int
    fp: int
    fn: int
    n: int
    direction: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FailureModeResult:
    points: list[FailureModePoint]
    threshold: float
    # error direction at the first sub-threshold SNR (clean->noisy), or None if
    # the model holds above threshold at every tested SNR.
    dominant_failure: str | None

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "dominant_failure": self.dominant_failure,
            "points": [p.to_dict() for p in self.points],
        }


def _counts_at_snr(
    model: StressNet, snr: float | None, n_per_class: int, seeds: Sequence[int]
) -> dict:
    """Summed confusion counts for ``model`` at one SNR across eval seeds.

    Counts are *paired and summed* across seeds (not averaged), so precision /
    recall / direction are computed once from a larger, more stable sample.
    """
    tp = tn = fp = fn = 0
    for s in seeds:
        x, y = noisy_synthetic_dataset(n_per_class, seed=s, snr_db=snr)
        with torch.no_grad():
            pred = model(x)
        c = _confusion(pred, y)
        tp += c["tp"]; tn += c["tn"]; fp += c["fp"]; fn += c["fn"]
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def noise_failure_mode(
    model: StressNet,
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_per_class: int = 64,
    seed: int = 1,
    eval_seeds: Sequence[int] | None = None,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> FailureModeResult:
    """Break the noise sweep down by error direction per SNR.

    ``eval_seeds`` (a list) sums confusion counts across seeds for a steadier
    breakdown; ``None`` (default) uses the single ``seed``. The clean point is
    first by convention so the scan runs least→most noise.
    """
    model = model.eval()
    seeds = list(eval_seeds) if eval_seeds is not None else [seed]

    points: list[FailureModePoint] = []
    for snr in snr_levels:
        c = _counts_at_snr(model, snr, n_per_class, seeds)
        tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
        n = tp + tn + fp + fn
        accuracy = (tp + tn) / max(1, n)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        points.append(
            FailureModePoint(
                snr_db=snr,
                accuracy=round(accuracy, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                tp=tp, tn=tn, fp=fp, fn=fn, n=n,
                direction=_direction(fp, fn),
            )
        )

    first_bad = next((p for p in points if p.accuracy < threshold), None)
    dominant = first_bad.direction if first_bad is not None else None
    out = FailureModeResult(points=points, threshold=threshold, dominant_failure=dominant)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "noise_failure_mode.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "noise_failure_mode.md").write_text(to_markdown(out))

    return out


_DIRECTION_GLOSS = {
    "balanced": "errors balanced (no dominant direction)",
    "misses_stress": "misses stress (false negatives — detector goes silent)",
    "false_alarms": "false alarms (false positives — detector cries wolf)",
}


def to_markdown(out: FailureModeResult) -> str:
    if out.dominant_failure is None:
        verdict = (
            f"**Holds above {out.threshold:.2f} at every tested SNR** — no "
            "dominant failure direction to mitigate."
        )
    else:
        verdict = (
            f"**First failure leans: {_DIRECTION_GLOSS[out.dominant_failure]}.** "
            "Tune the detector gate accordingly: a silent model wants a lower "
            "stress threshold / faster attack; a trigger-happy one wants a "
            "higher threshold / more smoothing."
        )
    header = (
        "# StressNet failure mode under noise\n\n"
        "Same waveform-noise SNR sweep as the robustness curve, split by error "
        "direction. `precision` falls when the model **false-alarms** (FP); "
        "`recall` falls when it **misses stress** (FN). The dominant direction "
        "is only named when one error type is at least "
        f"{_MARGIN:g}× the other.\n\n"
        f"- {verdict}\n\n"
        "| SNR | accuracy | precision | recall | FP | FN | direction |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {_snr_label(p.snr_db)} | {p.accuracy:.3f} | {p.precision:.3f} | "
        f"{p.recall:.3f} | {p.fp} | {p.fn} | {p.direction} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def build_noise_failure_mode(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    eval_n_per_class: int = 64,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> FailureModeResult:
    """Train the production model and record its noise failure-mode breakdown."""
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=threshold,
    )
    return noise_failure_mode(
        model, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=seed + 1, threshold=threshold, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="StressNet noise failure-mode breakdown")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_noise_failure_mode(
        epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, seed=args.seed, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/noise_failure_mode.json and noise_failure_mode.md")


if __name__ == "__main__":
    main()
