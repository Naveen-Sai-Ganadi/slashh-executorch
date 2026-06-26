"""Score-calibration analysis — are StressNet's sigmoid outputs real probabilities?

StressNet emits ``sigmoid(logit)`` in [0,1] and the live detector gates on
``STRESS_THRESHOLD = 0.6``. A threshold on a *score* is only meaningful if the
score is a calibrated *probability*: of the windows scored ~0.6, about 60%
should truly be stressed. If the model is over-confident, 0.6 means something
stronger than "60% sure" and the gate fires too eagerly (or too late).

This harness measures how well-calibrated the scores are over a noise sweep
(where scores actually spread out, unlike the near-deterministic clean set):

  * **ECE** (Expected Calibration Error) — bin by confidence, average the
    |confidence − accuracy| gap weighted by bin population.
  * **MCE** (Max Calibration Error) — the worst single bin.
  * **Brier score** — mean squared error of the probability itself.
  * a **reliability table** (per-bin confidence vs accuracy), and
  * one-parameter **temperature scaling** — the standard post-hoc fix — with
    the ECE it would achieve, so we know whether recalibration is worth it.

It then renders a verdict (well-calibrated / over- / under-confident). Either
outcome is shippable evidence: it tells us whether 0.6 is a trustworthy gate or
whether the scores need tempering first. Host-only; no device, no AI Hub token.

    python -m model.calibration                  # train, sweep, score, record
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .audio_config import STRESS_THRESHOLD
from .model import StressNet
from .robustness import noisy_synthetic_dataset

__all__ = [
    "CalibrationBin",
    "CalibrationResult",
    "compute_calibration",
    "collect_scores",
    "build_calibration",
    "to_markdown",
    "main",
]

# ECE at or below this is "well-calibrated" for our purposes (a common bar).
WELL_CALIBRATED_ECE = 0.05


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    confidence: float  # mean predicted confidence in the bin
    accuracy: float    # fraction correctly classified in the bin

    def to_dict(self) -> dict:
        return {
            "lo": self.lo, "hi": self.hi, "count": self.count,
            "confidence": self.confidence, "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class CalibrationResult:
    n: int
    n_bins: int
    ece: float
    mce: float
    brier: float
    accuracy: float
    signed_gap: float          # sum_bin w*(confidence - accuracy); >0 overconfident
    bins: list[CalibrationBin]
    temperature: float         # fitted T for temperature scaling (1.0 if not fit)
    ece_after_temp: float      # ECE achievable after temperature scaling
    threshold: float           # the product STRESS_THRESHOLD analyzed
    threshold_empirical_rate: float | None  # P(stress | score near threshold)
    verdict: str               # well-calibrated | overconfident | underconfident

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_bins": self.n_bins,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "signed_gap": self.signed_gap,
            "bins": [b.to_dict() for b in self.bins],
            "temperature": self.temperature,
            "ece_after_temp": self.ece_after_temp,
            "threshold": self.threshold,
            "threshold_empirical_rate": self.threshold_empirical_rate,
            "verdict": self.verdict,
        }


def _confidence_and_correct(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample (confidence in the predicted class, correct?) for binary scores."""
    pred = (scores >= 0.5).float()
    # confidence in the predicted class: p if predicting positive else 1-p
    conf = torch.where(pred == 1, scores, 1.0 - scores)
    correct = (pred == labels).float()
    return conf, correct


def _binned(conf: torch.Tensor, correct: torch.Tensor, n_bins: int) -> list[CalibrationBin]:
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # last bin is closed on the right so conf==1.0 lands somewhere
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
            continue
        bins.append(
            CalibrationBin(
                lo=lo, hi=hi, count=count,
                confidence=float(conf[mask].mean()),
                accuracy=float(correct[mask].mean()),
            )
        )
    return bins


def _ece_mce(bins: list[CalibrationBin], n: int) -> tuple[float, float, float]:
    """Return (ECE, MCE, signed_gap) from populated bins."""
    ece = mce = signed = 0.0
    for b in bins:
        if b.count == 0:
            continue
        w = b.count / n
        gap = b.confidence - b.accuracy
        ece += w * abs(gap)
        signed += w * gap
        mce = max(mce, abs(gap))
    return ece, mce, signed


def _fit_temperature(
    scores: torch.Tensor, labels: torch.Tensor, *, n_bins: int
) -> tuple[float, float]:
    """Grid-search a temperature T>0 that minimizes NLL; return (T, ECE_after).

    Scores are sigmoid probabilities; recover logits, scale by 1/T, re-sigmoid.
    A coarse-then-fine 1-D search is plenty for a single scalar and keeps the
    result deterministic (no optimizer RNG).
    """
    eps = 1e-6
    p = scores.clamp(eps, 1.0 - eps)
    logits = torch.log(p / (1.0 - p))
    y = labels

    def nll(T: float) -> float:
        q = torch.sigmoid(logits / T).clamp(eps, 1.0 - eps)
        return float(-(y * torch.log(q) + (1 - y) * torch.log(1 - q)).mean())

    best_T, best_nll = 1.0, nll(1.0)
    # coarse grid then refine around the best point
    grid = [0.5 + 0.1 * k for k in range(0, 86)]  # 0.5 .. 9.0
    for T in grid:
        v = nll(T)
        if v < best_nll:
            best_T, best_nll = T, v
    fine = [best_T + 0.01 * k for k in range(-9, 10) if best_T + 0.01 * k > 0]
    for T in fine:
        v = nll(T)
        if v < best_nll:
            best_T, best_nll = T, v

    q = torch.sigmoid(logits / best_T)
    conf, correct = _confidence_and_correct(q, y)
    ece_after, _, _ = _ece_mce(_binned(conf, correct, n_bins), int(y.numel()))
    return best_T, ece_after


def _threshold_rate(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float, *, half_width: float = 0.1
) -> float | None:
    """Empirical P(stress) among windows scored within ±half_width of threshold."""
    mask = (scores >= threshold - half_width) & (scores <= threshold + half_width)
    if int(mask.sum()) == 0:
        return None
    return float(labels[mask].mean())


def compute_calibration(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
    threshold: float = STRESS_THRESHOLD,
    fit_temperature: bool = True,
    out_dir: str | Path | None = None,
) -> CalibrationResult:
    """Reliability analysis of probability ``scores`` against binary ``labels``.

    ``scores`` are sigmoid outputs in [0,1]; ``labels`` are 0/1. Computes ECE,
    MCE, Brier, a per-bin reliability table, optional temperature scaling, and a
    verdict. Writes ``calibration.{json,md}`` into ``out_dir`` when given.
    """
    scores = scores.flatten().float()
    labels = labels.flatten().float()
    n = int(scores.numel())
    if n == 0:
        raise ValueError("need at least one (score, label) pair")

    conf, correct = _confidence_and_correct(scores, labels)
    bins = _binned(conf, correct, n_bins)
    ece, mce, signed = _ece_mce(bins, n)
    brier = float(((scores - labels) ** 2).mean())
    accuracy = float(correct.mean())

    if fit_temperature:
        temperature, ece_after = _fit_temperature(scores, labels, n_bins=n_bins)
    else:
        temperature, ece_after = 1.0, ece

    if ece <= WELL_CALIBRATED_ECE:
        verdict = "well-calibrated"
    elif signed > 0:
        verdict = "overconfident"
    else:
        verdict = "underconfident"

    result = CalibrationResult(
        n=n, n_bins=n_bins, ece=ece, mce=mce, brier=brier, accuracy=accuracy,
        signed_gap=signed, bins=bins, temperature=temperature,
        ece_after_temp=ece_after, threshold=threshold,
        threshold_empirical_rate=_threshold_rate(scores, labels, threshold),
        verdict=verdict,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "calibration.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        (out_dir / "calibration.md").write_text(to_markdown(result))

    return result


def collect_scores(
    model: StressNet,
    *,
    snr_levels: Sequence[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    n_per_class: int = 64,
    seed: int = 1,
    noise_color: str = "white",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool model scores + labels across a noise sweep into flat tensors.

    The clean set alone is near-deterministic (scores pinned at 0/1), so
    calibration is measured over the same SNR grid the robustness work uses —
    that is where the scores spread and the gate's reliability actually matters.
    """
    model = model.eval()
    all_scores, all_labels = [], []
    for snr in snr_levels:
        x, y = noisy_synthetic_dataset(
            n_per_class, seed=seed, snr_db=snr, noise_color=noise_color
        )
        with torch.no_grad():
            s = model(x).flatten()
        all_scores.append(s)
        all_labels.append(y.flatten())
    return torch.cat(all_scores), torch.cat(all_labels)


def _verdict_gloss(r: CalibrationResult) -> str:
    if r.verdict == "well-calibrated":
        return (
            f"scores are **well-calibrated** (ECE {r.ece:.3f} ≤ "
            f"{WELL_CALIBRATED_ECE:g}) — the {r.threshold:g} gate is a "
            "trustworthy probability threshold; no recalibration needed."
        )
    direction = "over" if r.signed_gap > 0 else "under"
    fix = (
        f"temperature scaling (T={r.temperature:.2f}) would cut ECE to "
        f"{r.ece_after_temp:.3f}"
        if r.temperature != 1.0
        else "consider temperature scaling"
    )
    return (
        f"scores are **{direction}-confident** (ECE {r.ece:.3f}, signed gap "
        f"{r.signed_gap:+.3f}) — the raw {r.threshold:g} gate is biased; "
        f"{fix}."
    )


def to_markdown(r: CalibrationResult) -> str:
    rate = (
        f"{r.threshold_empirical_rate:.3f}"
        if r.threshold_empirical_rate is not None
        else "—"
    )
    header = (
        "# Score calibration — are the sigmoid outputs real probabilities?\n\n"
        f"Over {r.n} windows (noise sweep): {_verdict_gloss(r)}\n\n"
        f"- **ECE** {r.ece:.4f}  ·  **MCE** {r.mce:.4f}  ·  **Brier** {r.brier:.4f}\n"
        f"- classification accuracy {r.accuracy:.3f}\n"
        f"- temperature scaling: **T={r.temperature:.3f}** → ECE "
        f"{r.ece_after_temp:.4f} (from {r.ece:.4f})\n"
        f"- empirical stress rate near the {r.threshold:g} gate "
        f"(±0.1): **{rate}**\n\n"
        "Reliability table (confidence in the predicted class vs realised "
        "accuracy per bin):\n\n"
        "| bin | n | mean confidence | accuracy | gap |\n|---|---|---|---|---|\n"
    )
    rows = []
    for b in r.bins:
        if b.count == 0:
            continue
        rows.append(
            f"| {b.lo:.1f}–{b.hi:.1f} | {b.count} | {b.confidence:.3f} | "
            f"{b.accuracy:.3f} | {b.confidence - b.accuracy:+.3f} |"
        )
    note = (
        "\n\n_Calibration is measured across the noise sweep, where scores "
        "spread; the clean set alone pins scores near 0/1 and is trivially "
        "calibrated. A positive gap = over-confident (the score claims more "
        "certainty than it earns)._\n"
    )
    return header + "\n".join(rows) + note


def build_calibration(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    eval_n_per_class: int = 96,
    snr_levels: Sequence[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    n_bins: int = 10,
    threshold: float = STRESS_THRESHOLD,
    out_dir: str | Path | None = "docs/benchmarks",
) -> CalibrationResult:
    """Train the production model, sweep noise, and record its calibration."""
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed, snr_levels=snr_levels,
    )
    scores, labels = collect_scores(
        model, snr_levels=snr_levels, n_per_class=eval_n_per_class, seed=seed + 1,
    )
    return compute_calibration(
        scores, labels, n_bins=n_bins, threshold=threshold, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score-calibration analysis")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = build_calibration(
        epochs=args.epochs, n_per_class=args.n_per_class, n_bins=args.n_bins,
        seed=args.seed, out_dir=args.out_dir,
    )
    print(to_markdown(r))
    print(f"wrote {args.out_dir}/calibration.json and calibration.md")


if __name__ == "__main__":
    main()
