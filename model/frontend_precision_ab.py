"""Front-end float32 vs float64 decision-parity A/B.

The deployable runs INT8 on the NPU, but the log-mel front-end
(``model/features.py``: ``MelSpectrogram`` then ``log10(mel + LOG_EPS)``) runs
in floating point on the CPU. The ``log10`` of near-zero mel energy in quiet
frames is the numerically delicate step: in principle single precision could
drift far enough there to push a borderline window's score across the stress
threshold, so the on-device float32 front-end would disagree with an
infinite-precision reference about a decision.

This A/B bounds that risk. It runs the SAME labelled noisy waveforms through the
front-end twice — once in float32, once in float64 — feeds both feature sets
into the same fp32 model, and measures per SNR (a) the worst score divergence
and (b) how many decisions flip across the threshold. Zero flips and a hair of
divergence certify the front-end is well-conditioned: the cheap single-precision
path the device actually runs loses nothing to numerics. Non-zero flips would
flag exactly which SNRs need a higher-precision front-end.

Host-only: pure CPU floating point, no device, no AI Hub token. The reduction is
pure (unit-tested without extraction); ``build_frontend_precision_ab`` trains one
tiny net and runs the front-end at both dtypes.

Run:
    PYTHONPATH=. .venv/bin/python -m model.frontend_precision_ab
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "FrontendPrecisionPoint",
    "FrontendPrecisionResult",
    "frontend_precision_ab",
    "build_frontend_precision_ab",
]

# A score divergence above this between float32 and float64 front-ends is "not a
# hair" — even with no decision flip it means the front-end is precision-sensitive
# enough to watch. 1e-3 is well below the hysteresis band the detector latches on.
_DEFAULT_TOL = 1e-3


@dataclass(frozen=True)
class FrontendPrecisionPoint:
    """Float32 vs float64 front-end agreement at one SNR (``None`` = clean)."""

    snr_db: float | None
    max_abs_diff: float  # worst |score_f32 - score_f64| over the SNR's windows
    n_flips: int         # windows whose threshold decision differs between dtypes
    n: int               # windows at this SNR

    @property
    def flip_rate(self) -> float:
        return self.n_flips / self.n

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flip_rate"] = self.flip_rate
        return d


@dataclass(frozen=True)
class FrontendPrecisionResult:
    """Front-end precision A/B across an SNR sweep."""

    points: list[FrontendPrecisionPoint]
    tol: float

    @property
    def worst_abs_diff(self) -> float:
        return max(p.max_abs_diff for p in self.points)

    @property
    def total_flips(self) -> int:
        return sum(p.n_flips for p in self.points)

    @property
    def total_n(self) -> int:
        return sum(p.n for p in self.points)

    @property
    def flip_rate(self) -> float:
        return self.total_flips / self.total_n

    @property
    def precision_robust(self) -> bool:
        """True if single precision changes no decision and barely any score."""
        return self.total_flips == 0 and self.worst_abs_diff <= self.tol

    @property
    def verdict(self) -> str:
        if self.precision_robust:
            return (
                "**Front-end is precision-robust**: float32 matches float64 to "
                f"within {self.worst_abs_diff:.2e} with **zero** decision flips "
                f"across {self.total_n} windows. The on-device single-precision "
                "log-mel front-end loses nothing to numerics — no need for a "
                "higher-precision path."
            )
        if self.total_flips > 0:
            worst = max(self.points, key=lambda p: p.n_flips)
            return (
                f"**Precision matters**: {self.total_flips}/{self.total_n} "
                f"decisions flip between float32 and float64 (worst at "
                f"{_snr_label(worst.snr_db)}: {worst.n_flips}/{worst.n}). The "
                "front-end is borderline-sensitive; consider a float64 log-mel for "
                "windows near the threshold."
            )
        return (
            "**No flips, but watch it**: decisions agree, yet float32 diverges "
            f"from float64 by up to {self.worst_abs_diff:.2e} (over the "
            f"{self.tol:.0e} tolerance). Conditioning is adequate today but the "
            "margin is thinner than ideal."
        )

    def to_dict(self) -> dict:
        return {
            "tol": self.tol,
            "worst_abs_diff": self.worst_abs_diff,
            "total_flips": self.total_flips,
            "total_n": self.total_n,
            "flip_rate": self.flip_rate,
            "precision_robust": self.precision_robust,
            "verdict": self.verdict,
            "points": [p.to_dict() for p in self.points],
        }


def frontend_precision_ab(
    records,
    *,
    tol: float = _DEFAULT_TOL,
    out_dir: str | Path | None = None,
) -> FrontendPrecisionResult:
    """Reduce per-SNR float32-vs-float64 front-end records to a verdict.

    ``records`` is an iterable of ``(snr_db, max_abs_diff, n_flips, n)`` tuples,
    clean->noisy. Pure — no extraction, no device. Each ``n`` must be positive.
    Writes ``frontend_precision_ab.{json,md}`` to ``out_dir`` when given.
    """
    points = [
        FrontendPrecisionPoint(
            snr_db=s, max_abs_diff=float(d), n_flips=int(f), n=int(n)
        )
        for s, d, f, n in records
    ]
    if not points:
        raise ValueError("records must be non-empty")
    if any(p.n <= 0 for p in points):
        raise ValueError("each n must be positive")

    out = FrontendPrecisionResult(points=points, tol=tol)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frontend_precision_ab.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "frontend_precision_ab.md").write_text(to_markdown(out))

    return out


def _snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def to_markdown(out: FrontendPrecisionResult) -> str:
    header = (
        "# Front-end float32 vs float64 parity (A/B)\n\n"
        "The model runs INT8 on the NPU, but the log-mel front-end runs in "
        "floating point on the CPU. The same labelled noisy waveforms go through "
        "the front-end in **float32** and **float64**, both feature sets feed the "
        "same fp32 model, and each row reports the worst score divergence and how "
        "many threshold *decisions* flip — a certification that the cheap "
        "single-precision path the device runs is well-conditioned.\n\n"
        f"- worst |Δ score| (f32 vs f64): **{out.worst_abs_diff:.2e}** "
        f"(tolerance {out.tol:.0e})\n"
        f"- decision flips: **{out.total_flips}** / {out.total_n} windows "
        f"(flip rate {out.flip_rate:.4f})\n"
        f"- precision-robust: **{out.precision_robust}**\n"
        f"- {out.verdict}\n\n"
        "| SNR | max |Δ score| | flips | n | flip rate |\n"
        "|---|---|---|---|---|\n"
    )
    rows = [
        f"| {_snr_label(p.snr_db)} | {p.max_abs_diff:.2e} | {p.n_flips} | {p.n} | "
        f"{p.flip_rate:.4f} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def _extract_at(wave, *, double: bool):
    """Run the production log-mel front-end on one waveform at a chosen dtype.

    Mirrors ``features.extract`` (pad/truncate to N_FRAMES) but builds a fresh
    extractor cast to float32 or float64 so the whole mel + log10 pipeline runs
    at that precision.
    """
    import torch

    from .audio_config import N_FRAMES, N_MELS
    from .features import LogMelExtractor

    extractor = LogMelExtractor().eval()
    if double:
        extractor = extractor.double()
        wave = wave.double()
    else:
        extractor = extractor.float()
        wave = wave.float()
    with torch.no_grad():
        logmel = extractor(wave)
    frames = logmel.shape[-1]
    if frames < N_FRAMES:
        logmel = torch.nn.functional.pad(logmel, (0, N_FRAMES - frames))
    elif frames > N_FRAMES:
        logmel = logmel[..., :N_FRAMES]
    return logmel.reshape(1, 1, N_MELS, N_FRAMES)


def build_frontend_precision_ab(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.5,
    tol: float = _DEFAULT_TOL,
    out_dir: str | Path | None = "docs/benchmarks",
) -> FrontendPrecisionResult:
    """Train one net, run the front-end at float32 + float64, compare decisions.

    The model is the shipped fp32 production net; the only thing that varies
    between the two feature sets is the front-end arithmetic precision. Imports
    inside the function to keep torch/training off the pure reduction path.
    """
    import torch

    from .data import _synth_waveform
    from .production import train_production
    from .robustness import _add_noise

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=0.8,
    )
    model = model.eval()

    records = []
    for snr in snr_levels:
        gen = torch.Generator().manual_seed(seed + 1)
        feats32, feats64 = [], []
        for stressed in (False, True):
            for _ in range(eval_n_per_class):
                wave = _synth_waveform(stressed, gen)
                if snr is not None:
                    wave = _add_noise(wave, snr, gen)
                feats32.append(_extract_at(wave, double=False))
                feats64.append(_extract_at(wave, double=True))
        x32 = torch.cat(feats32, dim=0).float()
        x64 = torch.cat(feats64, dim=0).float()  # model is fp32; cast features in
        with torch.no_grad():
            s32 = model(x32).flatten()
            s64 = model(x64).flatten()
        diff = (s32 - s64).abs()
        flips = ((s32 >= threshold) != (s64 >= threshold)).sum().item()
        records.append((snr, float(diff.max()), int(flips), int(x32.shape[0])))

    return frontend_precision_ab(records, tol=tol, out_dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Front-end float32 vs float64 decision-parity A/B"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_frontend_precision_ab(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/frontend_precision_ab.json and .md")


if __name__ == "__main__":
    main()
