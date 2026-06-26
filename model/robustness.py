"""Noise-robustness sweep for StressNet.

Clean-set accuracy (model/eval.py) is necessary but not sufficient: on-device
the classifier hears real-world audio with traffic, fans, crowd, and mic hiss.
This sweep injects additive white noise **on the waveform** — before the
log-mel extractor, where physical acoustic noise actually lives — at a range of
SNRs, and reports how accuracy degrades. The result is an accuracy-vs-SNR curve
and an "operating floor": the SNR below which the model stops being reliable.

    python -m model.robustness --weights assets/stress_model.pt
    python -m model.robustness            # trains a quick model, then sweeps

That floor informs the VAD gate and the EMA smoothing (model/detector.py): if
the model is shaky below, say, 10 dB SNR, the pipeline should lean on smoothing
and require cleaner windows before reacting. Host-only; no device, no AI Hub.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .audio_config import SAMPLE_RATE, WINDOW_SAMPLES
from .data import _synth_waveform
from .eval import _confusion
from .features import extract
from .model import StressNet, build_model

# Re-exported so callers don't reach into model.data for the clean generator.
__all__ = [
    "RobustnessPoint",
    "RobustnessResult",
    "noisy_synthetic_dataset",
    "robustness_curve",
    "operating_floor",
    "reliable_floor",
]


def _add_noise(wave: torch.Tensor, snr_db: float, gen: torch.Generator) -> torch.Tensor:
    """Add white Gaussian noise to ``wave`` at the requested SNR (dB)."""
    sig_power = wave.pow(2).mean().clamp(min=1e-12)
    snr = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr
    noise = torch.randn(wave.shape, generator=gen) * noise_power.sqrt()
    return wave + noise


def noisy_synthetic_dataset(
    n_per_class: int, seed: int = 0, snr_db: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced synthetic eval set with optional waveform noise at ``snr_db``.

    ``snr_db=None`` injects no noise and reproduces ``synthetic_dataset`` exactly
    (same generator draws), so the clean point is directly comparable.
    """
    gen = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for stressed in (False, True):
        for _ in range(n_per_class):
            wave = _synth_waveform(stressed, gen)
            if snr_db is not None:
                wave = _add_noise(wave, snr_db, gen)
            feats.append(extract(wave))
            labels.append(float(stressed))
    x = torch.cat(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


@dataclass(frozen=True)
class RobustnessPoint:
    """Accuracy / F1 of the model at one SNR (``snr_db=None`` = clean)."""

    snr_db: float | None
    accuracy: float
    f1: float
    n: int
    # std of accuracy across eval seeds when the curve is averaged over several;
    # None for a single-seed curve. Default keeps back-compat with old records.
    acc_std: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessResult:
    points: list[RobustnessPoint]
    floor_db: float | None  # first SNR (clean->noisy) below the accuracy threshold
    # lowest SNR still at/above threshold — the noise the model is reliable *down
    # to*. Defaults None for back-compat with records written before it existed.
    reliable_floor_db: float | None = None

    def to_dict(self) -> dict:
        return {
            "floor_db": self.floor_db,
            "reliable_floor_db": self.reliable_floor_db,
            "points": [p.to_dict() for p in self.points],
        }


def operating_floor(
    points: list[RobustnessPoint], *, threshold: float = 0.8
) -> float | None:
    """First SNR (scanning clean→noisy) whose accuracy drops below ``threshold``.

    Returns the ``snr_db`` of that point, or ``None`` if every point holds up.
    """
    for p in points:
        if p.accuracy < threshold:
            return p.snr_db
    return None


def reliable_floor(
    points: list[RobustnessPoint], *, threshold: float = 0.8
) -> float | None:
    """Lowest numeric SNR (scanning clean→noisy) still at/above ``threshold``.

    This is the noise level the model is reliable *down to* — the last point
    before :func:`operating_floor`'s first failure. It is the honest number to
    advertise as the "floor": for a curve that holds to 0 dB and only collapses
    at -5 dB, this returns ``0.0`` while ``operating_floor`` returns ``-5.0``.
    Reporting the failing SNR alone (and printing it beside its own
    sub-threshold accuracy row) reads as a contradiction; this pairs with it.

    Returns ``None`` if the model isn't reliable at any *noisy* SNR (only clean,
    or not even clean) — the clean point (``snr_db=None``) is never a floor.
    """
    floor: float | None = None
    for p in points:
        if p.accuracy < threshold:
            break
        if p.snr_db is not None:
            floor = p.snr_db
    return floor


def _eval_one_seed(
    model: StressNet, snr_levels, n_per_class: int, seed: int
) -> list[dict]:
    """Per-SNR accuracy/f1/n for ``model`` at one eval seed."""
    rows = []
    for snr in snr_levels:
        x, y = noisy_synthetic_dataset(n_per_class, seed=seed, snr_db=snr)
        with torch.no_grad():
            pred = model(x)
        m = _confusion(pred, y)
        rows.append({"accuracy": m["accuracy"], "f1": m["f1"], "n": int(x.shape[0])})
    return rows


def robustness_curve(
    model: StressNet,
    *,
    snr_levels: list[float | None] = (None, 30.0, 20.0, 10.0, 0.0, -10.0),
    n_per_class: int = 64,
    seed: int = 1,
    eval_seeds: Sequence[int] | None = None,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> RobustnessResult:
    """Evaluate ``model`` across noise levels; optionally write artifacts.

    The clean point is first by convention so ``operating_floor`` scans from
    least to most noise. With ``eval_seeds`` (a list), each SNR is evaluated at
    every seed and the curve reports the **mean** accuracy/f1 with the accuracy
    **std** per point — stronger, less luck-dependent evidence than a single
    draw. ``eval_seeds=None`` (default) keeps the single-``seed`` behaviour and
    reports no std (``acc_std=None``).
    """
    model = model.eval()
    seeds = list(eval_seeds) if eval_seeds is not None else [seed]
    per_seed = [_eval_one_seed(model, snr_levels, n_per_class, s) for s in seeds]

    points: list[RobustnessPoint] = []
    for j, snr in enumerate(snr_levels):
        accs = [rows[j]["accuracy"] for rows in per_seed]
        f1s = [rows[j]["f1"] for rows in per_seed]
        std = round(statistics.stdev(accs), 4) if len(accs) > 1 else None
        points.append(
            RobustnessPoint(
                snr_db=snr,
                accuracy=round(statistics.fmean(accs), 4),
                f1=round(statistics.fmean(f1s), 4),
                n=per_seed[0][j]["n"],
                acc_std=std,
            )
        )

    floor = operating_floor(points, threshold=threshold)
    reliable = reliable_floor(points, threshold=threshold)
    out = RobustnessResult(points=points, floor_db=floor, reliable_floor_db=reliable)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "robustness.json").write_text(json.dumps(out.to_dict(), indent=2) + "\n")
        (out_dir / "robustness.md").write_text(to_markdown(out, threshold=threshold))

    return out


def _snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def to_markdown(out: RobustnessResult, *, threshold: float = 0.8) -> str:
    """Render the accuracy-vs-SNR curve as a table."""
    reliable = (
        "none (not reliable below clean)"
        if out.reliable_floor_db is None
        else _snr_label(out.reliable_floor_db)
    )
    drops = (
        "never (holds at all tested SNRs)"
        if out.floor_db is None
        else _snr_label(out.floor_db)
    )
    header = (
        "# StressNet noise robustness\n\n"
        "White Gaussian noise is added to each waveform at the SNR below, then "
        "features are re-extracted and the model evaluated. The model is "
        f"reliable down to the lowest SNR where accuracy ≥ {threshold:.2f}, and "
        "drops below at the next, noisier level.\n\n"
        f"- **reliable down to: {reliable}** (accuracy ≥ {threshold:.2f})\n"
        f"- drops below {threshold:.2f} at: {drops}\n\n"
        "| SNR | accuracy | f1 | n |\n"
        "|---|---|---|---|\n"
    )
    rows = [
        f"| {_snr_label(p.snr_db)} | {p.accuracy:.3f} | {p.f1:.3f} | {p.n} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="StressNet noise-robustness sweep")
    ap.add_argument("--weights", "-w", default=None,
                    help="checkpoint to evaluate; omit to train a quick model")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    if args.weights:
        model = build_model(args.weights)
    else:
        from .train import train

        print("no --weights given; training a quick model for the sweep...")
        model, _ = train(data_dir=None, epochs=12, batch_size=16, lr=1e-3,
                         n_per_class=96, seed=0)

    out = robustness_curve(
        model, n_per_class=args.n_per_class, seed=args.seed,
        threshold=args.threshold, out_dir=args.out_dir,
    )
    print(to_markdown(out, threshold=args.threshold))
    print(f"wrote {args.out_dir}/robustness.json and robustness.md")


if __name__ == "__main__":
    main()
