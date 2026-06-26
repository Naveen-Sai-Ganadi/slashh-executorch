"""Operating-point / ROC sweep A/B.

The per-window classifier emits a probability and thresholds it at a fixed 0.5 to
decide stressed/calm — the raw decision the detector's EMA + hysteresis then
smooth. Two existing artifacts touch the decision boundary but neither sweeps the
*per-window* threshold itself: ``tune_detector`` tunes the hysteresis band /
smoothing, and ``base_rate_precision`` evaluates TPR/FPR at the single 0.5
threshold then sweeps *prevalence*. This A/B fills the gap with the classic
operating-point analysis.

It sweeps the decision threshold across ``[0, 1]``, computes TPR / FPR /
accuracy / F1 / Youden's J at each, integrates the ROC AUC (trapezoidal over the
swept points), and finds the accuracy-, F1-, and Youden-optimal thresholds. It
then reports how far the default 0.5 sits from the accuracy optimum — the
accuracy *headroom* left on the table — and whether 0.5 is well placed.

Host-only: CPU training + the pure-Python front-end, no device, no AI Hub token.
The reduction is pure (unit-tested with hand-set ROC points);
``build_operating_point`` trains one tiny net and sweeps real scores.

Run:
    PYTHONPATH=. .venv/bin/python -m model.operating_point
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OperatingPoint",
    "OperatingPointResult",
    "operating_point",
    "build_operating_point",
]

_DEFAULT_THRESHOLD = 0.5
# Accuracy gap (default vs best) within which 0.5 is deemed "well placed".
_DEFAULT_HEADROOM_TOL = 0.02


@dataclass(frozen=True)
class OperatingPoint:
    """Classifier metrics at one decision threshold."""

    threshold: float
    tpr: float  # sensitivity / recall on the positive (stressed) class
    fpr: float  # 1 - specificity
    accuracy: float
    f1: float
    n_pos: int
    n_neg: int

    @property
    def youden_j(self) -> float:
        """Youden's J = TPR - FPR (informedness; maximized at the ROC knee)."""
        return self.tpr - self.fpr

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "youden_j": self.youden_j,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }


@dataclass(frozen=True)
class OperatingPointResult:
    """ROC / operating-point reduction over a decision-threshold sweep."""

    points: list[OperatingPoint]  # sorted by threshold ascending
    default_threshold: float
    headroom_tol: float

    @property
    def auc(self) -> float:
        """ROC AUC, trapezoidal over the swept points (sorted by FPR ascending).

        Integrates TPR d(FPR). With a perfect classifier present (a point at
        TPR=1, FPR=0 spanning to FPR=1) this approaches 1.0; the diagonal gives
        0.5. It reflects the thresholds actually swept — coarser sweeps integrate
        a coarser curve.
        """
        roc = sorted(((p.fpr, p.tpr) for p in self.points), key=lambda t: (t[0], t[1]))
        area = 0.0
        for (x0, y0), (x1, y1) in zip(roc, roc[1:]):
            area += (x1 - x0) * (y0 + y1) / 2.0
        return area

    @property
    def best_accuracy(self) -> float:
        return max(p.accuracy for p in self.points)

    @property
    def best_accuracy_threshold(self) -> float:
        return max(self.points, key=lambda p: p.accuracy).threshold

    @property
    def best_f1(self) -> float:
        return max(p.f1 for p in self.points)

    @property
    def best_f1_threshold(self) -> float:
        return max(self.points, key=lambda p: p.f1).threshold

    @property
    def best_youden_j(self) -> float:
        return max(p.youden_j for p in self.points)

    @property
    def best_youden_threshold(self) -> float:
        return max(self.points, key=lambda p: p.youden_j).threshold

    @property
    def _default_point(self) -> OperatingPoint:
        """Point at the default threshold, or the nearest swept threshold."""
        return min(self.points, key=lambda p: abs(p.threshold - self.default_threshold))

    @property
    def default_accuracy(self) -> float:
        return self._default_point.accuracy

    @property
    def accuracy_headroom(self) -> float:
        """How much accuracy a better threshold would buy over the default (>= 0)."""
        return self.best_accuracy - self.default_accuracy

    @property
    def well_placed(self) -> bool:
        """Default 0.5 is within ``headroom_tol`` accuracy of the optimum."""
        return self.accuracy_headroom <= self.headroom_tol

    @property
    def verdict(self) -> str:
        dp = self._default_point
        if self.well_placed:
            return (
                f"**Default 0.5 is well placed**: AUC {self.auc:.3f}; the default "
                f"per-window threshold yields {dp.accuracy:.0%} accuracy, within "
                f"{self.accuracy_headroom:.1%} of the best ({self.best_accuracy:.0%} "
                f"at {self.best_accuracy_threshold:g}). No retune of the raw "
                "decision boundary is warranted — tune the hysteresis band instead."
            )
        return (
            f"**Threshold headroom available**: AUC {self.auc:.3f}; the default 0.5 "
            f"gives {dp.accuracy:.0%} accuracy but moving the per-window threshold "
            f"to {self.best_accuracy_threshold:g} reaches {self.best_accuracy:.0%} "
            f"(+{self.accuracy_headroom:.1%}). Youden-optimal is "
            f"{self.best_youden_threshold:g} (J={self.best_youden_j:.2f}), F1-optimal "
            f"{self.best_f1_threshold:g}. Consider shifting the raw decision "
            "threshold before it feeds the EMA + hysteresis stage."
        )

    def to_dict(self) -> dict:
        return {
            "default_threshold": self.default_threshold,
            "headroom_tol": self.headroom_tol,
            "n_points": len(self.points),
            "auc": self.auc,
            "best_accuracy": self.best_accuracy,
            "best_accuracy_threshold": self.best_accuracy_threshold,
            "best_f1": self.best_f1,
            "best_f1_threshold": self.best_f1_threshold,
            "best_youden_j": self.best_youden_j,
            "best_youden_threshold": self.best_youden_threshold,
            "default_accuracy": self.default_accuracy,
            "accuracy_headroom": self.accuracy_headroom,
            "well_placed": self.well_placed,
            "verdict": self.verdict,
            "points": [p.to_dict() for p in self.points],
        }


def operating_point(
    records,
    *,
    default_threshold: float = _DEFAULT_THRESHOLD,
    headroom_tol: float = _DEFAULT_HEADROOM_TOL,
    out_dir: str | Path | None = None,
) -> OperatingPointResult:
    """Reduce per-threshold ``(threshold, tpr, fpr, accuracy, f1, n_pos, n_neg)`` records.

    Points are sorted by ``threshold``. Pure — no training, no device. All rates
    must be in ``[0, 1]``, thresholds in ``[0, 1]``, and ``n_pos``/``n_neg``
    positive. Writes ``operating_point.{json,md}`` to ``out_dir`` when given.
    """
    points = []
    for threshold, tpr, fpr, acc, f1, n_pos, n_neg in records:
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        for name, v in (("tpr", tpr), ("fpr", fpr), ("accuracy", acc), ("f1", f1)):
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if int(n_pos) <= 0 or int(n_neg) <= 0:
            raise ValueError("n_pos and n_neg must be positive")
        points.append(OperatingPoint(
            threshold=float(threshold), tpr=float(tpr), fpr=float(fpr),
            accuracy=float(acc), f1=float(f1), n_pos=int(n_pos), n_neg=int(n_neg),
        ))
    if not points:
        raise ValueError("records must be non-empty")
    points.sort(key=lambda p: p.threshold)

    out = OperatingPointResult(
        points=points, default_threshold=default_threshold, headroom_tol=headroom_tol,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "operating_point.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "operating_point.md").write_text(to_markdown(out))

    return out


def to_markdown(out: OperatingPointResult) -> str:
    header = (
        "# Operating-point / ROC sweep (A/B)\n\n"
        "The per-window classifier thresholds its probability at a fixed 0.5 before "
        "the EMA + hysteresis smoothing. This sweeps that decision threshold, "
        "computes TPR / FPR / accuracy / F1 / Youden's J at each, integrates the "
        "ROC AUC, and asks whether 0.5 is the right operating point.\n\n"
        f"- ROC AUC (trapezoidal): **{out.auc:.3f}**\n"
        f"- default ({out.default_threshold:g}) accuracy: **{out.default_accuracy:.0%}**\n"
        f"- best accuracy: **{out.best_accuracy:.0%}** at threshold "
        f"**{out.best_accuracy_threshold:g}** (headroom **+{out.accuracy_headroom:.1%}**)\n"
        f"- Youden-optimal threshold: **{out.best_youden_threshold:g}** "
        f"(J={out.best_youden_j:.2f})\n"
        f"- F1-optimal threshold: **{out.best_f1_threshold:g}** (F1={out.best_f1:.0%})\n"
        f"- 0.5 well placed (tol {out.headroom_tol:.0%}): **{out.well_placed}**\n"
        f"- {out.verdict}\n\n"
        "| threshold | TPR | FPR | accuracy | F1 | Youden J |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {p.threshold:g} | {p.tpr:.0%} | {p.fpr:.0%} | {p.accuracy:.0%} "
        f"| {p.f1:.0%} | {p.youden_j:+.2f} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def build_operating_point(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    eval_snr_levels=(None, 10.0, 0.0, -5.0),
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    default_threshold: float = _DEFAULT_THRESHOLD,
    headroom_tol: float = _DEFAULT_HEADROOM_TOL,
    out_dir: str | Path | None = "docs/benchmarks",
) -> OperatingPointResult:
    """Train one net, sweep the per-window decision threshold over real scores.

    A balanced eval set is **pooled across ``eval_snr_levels``** (clean + a spread
    of noisy SNRs) so the ROC reflects the realistic class overlap the detector
    sees in the field, not the trivially-separable clean case. It is scored once;
    each threshold then partitions the same probabilities into TP/FP/TN/FN and
    yields TPR/FPR/accuracy/F1. Imports inside the function keep torch/training off
    the pure reduction path.
    """
    import torch

    from .data import _synth_waveform
    from .features import extract
    from .production import train_production
    from .robustness import _add_noise

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=0.8,
    )
    model = model.eval()

    gen = torch.Generator().manual_seed(seed + 7)
    feats, labels = [], []
    for snr in eval_snr_levels:
        for stressed in (False, True):
            for _ in range(eval_n_per_class):
                wave = _synth_waveform(stressed, gen)
                if snr is not None:
                    wave = _add_noise(wave, snr, gen, color="white")
                feats.append(extract(wave))
                labels.append(float(stressed))
    x = torch.cat(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).flatten()

    pos = y == 1.0
    neg = y == 0.0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())

    records = []
    for thr in thresholds:
        pred = (scores >= thr).float()
        tp = float((pred[pos] == 1.0).sum().item())
        fn = n_pos - tp
        fp = float((pred[neg] == 1.0).sum().item())
        tn = n_neg - fp
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        acc = (tp + tn) / (n_pos + n_neg)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        records.append((thr, tpr, fpr, acc, f1, n_pos, n_neg))

    return operating_point(
        records, default_threshold=default_threshold,
        headroom_tol=headroom_tol, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Operating-point / ROC sweep A/B")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_operating_point(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/operating_point.json and .md")


if __name__ == "__main__":
    main()
