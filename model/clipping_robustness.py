"""Clipping / saturation robustness A/B.

``gain_robustness`` swept input *level* with pure linear scaling — no clipping —
and found the loud end tolerated to +36 dB. But that model is incomplete for the
loud end: a real microphone / ADC **clips** at full scale. Loud or close-talking
capture flat-tops the waveform, a *nonlinear* distortion (harmonics, lost peaks)
that neither additive-noise SNR (``robustness``) nor linear gain (``gain_robustness``)
reproduces. This A/B isolates clipping.

For each ``clip_ratio`` the waveform is hard-clipped to ``clip_ratio * peak`` (a
fraction of its own peak amplitude) and then **rescaled back to the original
peak**, so the absolute level is unchanged and only the flat-topping distortion
remains — clipping is separated cleanly from the level axis ``gain_robustness``
already owns. It re-extracts log-mel features from the clipped waveforms and
measures accuracy, reporting the most aggressive clipping the model tolerates
before saturation breaks detection.

``clip_ratio`` = fraction of peak amplitude retained before clamping: 1.0 = no
clip, 0.1 = clamp to 10% of peak (heavy saturation). ``clipped_fraction`` = the
share of samples that actually hit the clip, reported for context.

Host-only: CPU training + the pure-Python front-end, no device, no AI Hub token.
The reduction is pure (unit-tested with hand-set accuracies);
``build_clipping_robustness`` trains one tiny net and evaluates clipped eval sets.

Run:
    PYTHONPATH=. .venv/bin/python -m model.clipping_robustness
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ClipPoint",
    "ClippingRobustnessResult",
    "clipping_robustness",
    "build_clipping_robustness",
]

# Project-wide "reliable" accuracy convention (matches robustness threshold).
_DEFAULT_ACCURACY_BAR = 0.8


@dataclass(frozen=True)
class ClipPoint:
    """Accuracy at one clip severity.

    ``clip_ratio`` is the fraction of peak amplitude retained before clamping
    (1.0 = unclipped). ``clipped_fraction`` is the share of samples that hit the
    clip, informational.
    """

    clip_ratio: float
    accuracy: float
    clipped_fraction: float
    n: int

    def reliable(self, bar: float) -> bool:
        return self.accuracy >= bar

    def to_dict(self) -> dict:
        return {
            "clip_ratio": self.clip_ratio,
            "accuracy": self.accuracy,
            "clipped_fraction": self.clipped_fraction,
            "n": self.n,
        }


@dataclass(frozen=True)
class ClippingRobustnessResult:
    """Clipping-robustness reduction over a clip-severity sweep."""

    points: list[ClipPoint]  # sorted by clip_ratio descending (clean first)
    accuracy_bar: float

    @property
    def _clean(self) -> ClipPoint | None:
        for p in self.points:
            if p.clip_ratio == 1.0:
                return p
        return None

    @property
    def clean_accuracy(self) -> float | None:
        c = self._clean
        return None if c is None else c.accuracy

    @property
    def worst_accuracy(self) -> float:
        return min(p.accuracy for p in self.points)

    @property
    def worst_clip_ratio(self) -> float:
        return min(self.points, key=lambda p: p.accuracy).clip_ratio

    @property
    def clip_tolerant(self) -> bool:
        """Every point in the sweep clears the bar."""
        return all(p.reliable(self.accuracy_bar) for p in self.points)

    @property
    def reliable_floor_ratio(self) -> float | None:
        """Lowest ``clip_ratio`` still reliable, contiguous from the clean (1.0) end.

        Points are ordered clean→severe (clip_ratio descending). Walk down from
        the unclipped reference while each successive point clears the bar; the
        last such ratio is the most aggressive clipping tolerated. ``None`` if
        there is no unclipped reference or it already fails the bar.
        """
        c = self._clean
        if c is None or not c.reliable(self.accuracy_bar):
            return None
        floor = c.clip_ratio
        idx = self.points.index(c)
        i = idx
        while i + 1 < len(self.points) and self.points[i + 1].reliable(self.accuracy_bar):
            i += 1
            floor = self.points[i].clip_ratio
        return floor

    @property
    def verdict(self) -> str:
        if self.clip_tolerant:
            return (
                "**Clip-tolerant**: detection clears the "
                f"{self.accuracy_bar:.0%} bar across the entire clip sweep "
                f"(worst {self.worst_accuracy:.0%} at clip_ratio {self.worst_clip_ratio:g}). "
                "Flat-topping from a saturated mic / ADC doesn't break detection, "
                "so no anti-clipping headroom is required at the input."
            )
        if self._clean is None or not self._clean.reliable(self.accuracy_bar):
            return (
                "**Fails unclipped**: accuracy is below the "
                f"{self.accuracy_bar:.0%} bar even at clip_ratio 1.0 (no clipping) — "
                "the clip sweep can't be characterized around a broken operating "
                "point. Re-check the model before tuning clip tolerance."
            )
        floor = self.reliable_floor_ratio
        return (
            "**Reliable down to a clip floor**: detection holds above the "
            f"{self.accuracy_bar:.0%} bar only while clip_ratio stays at or above "
            f"**{floor:g}** (clamping to {floor:.0%} of peak); below that, saturation "
            f"distortion drops accuracy to {self.worst_accuracy:.0%} at clip_ratio "
            f"{self.worst_clip_ratio:g}. Loud or close-talking capture that flat-tops "
            "harder than this will be missed — add input headroom / anti-clip "
            "limiting, or train with clipped augmentation to widen tolerance."
        )

    def to_dict(self) -> dict:
        return {
            "accuracy_bar": self.accuracy_bar,
            "n_points": len(self.points),
            "clean_accuracy": self.clean_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "worst_clip_ratio": self.worst_clip_ratio,
            "clip_tolerant": self.clip_tolerant,
            "reliable_floor_ratio": self.reliable_floor_ratio,
            "verdict": self.verdict,
            "points": [p.to_dict() for p in self.points],
        }


def clipping_robustness(
    records,
    *,
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = None,
) -> ClippingRobustnessResult:
    """Reduce per-severity ``(clip_ratio, accuracy, clipped_fraction, n)`` records.

    Points are sorted by ``clip_ratio`` descending (clean first). Pure — no
    training, no device. ``clip_ratio`` must be in ``(0, 1]``, ``accuracy`` and
    ``clipped_fraction`` in ``[0, 1]``, and ``n`` positive. Writes
    ``clipping_robustness.{json,md}`` to ``out_dir`` when given.
    """
    points = []
    for clip_ratio, acc, clipped_frac, n in records:
        if not (0.0 < float(clip_ratio) <= 1.0):
            raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")
        if not (0.0 <= float(acc) <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {acc}")
        if not (0.0 <= float(clipped_frac) <= 1.0):
            raise ValueError(f"clipped_fraction must be in [0, 1], got {clipped_frac}")
        if int(n) <= 0:
            raise ValueError("n must be positive")
        points.append(ClipPoint(
            clip_ratio=float(clip_ratio), accuracy=float(acc),
            clipped_fraction=float(clipped_frac), n=int(n),
        ))
    if not points:
        raise ValueError("records must be non-empty")
    points.sort(key=lambda p: p.clip_ratio, reverse=True)

    out = ClippingRobustnessResult(points=points, accuracy_bar=accuracy_bar)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "clipping_robustness.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "clipping_robustness.md").write_text(to_markdown(out))

    return out


def _fmt(x: float | None, unit: str = "") -> str:
    return "—" if x is None else f"{x:g}{unit}"


def to_markdown(out: ClippingRobustnessResult) -> str:
    floor = out.reliable_floor_ratio
    floor_txt = (
        f"{floor:g} (clamp to {floor:.0%} of peak)" if floor is not None else "—"
    )
    header = (
        "# Clipping / saturation robustness (A/B)\n\n"
        "`gain_robustness` swept input level with linear scaling; this isolates "
        "the *nonlinear* loud-end failure a real mic/ADC produces: clipping. Each "
        "waveform is hard-clipped to `clip_ratio * peak`, then rescaled back to the "
        "original peak (level held fixed, only flat-topping remains), re-extracted "
        "to log-mel, and scored — the most aggressive clipping the model tolerates "
        "before saturation breaks detection.\n\n"
        f"- unclipped (clip_ratio 1.0) accuracy: **{_fmt(out.clean_accuracy)}**\n"
        f"- reliable clip floor (bar {out.accuracy_bar:.0%}): **{floor_txt}**\n"
        f"- clip-tolerant across sweep: **{out.clip_tolerant}**\n"
        f"- worst: **{out.worst_accuracy:.0%}** at **clip_ratio {out.worst_clip_ratio:g}**\n"
        f"- {out.verdict}\n\n"
        "| clip_ratio | clipped samples | accuracy | reliable |\n"
        "|---|---|---|---|\n"
    )
    rows = [
        f"| {p.clip_ratio:g} | {p.clipped_fraction:.0%} | {p.accuracy:.0%} "
        f"| {p.reliable(out.accuracy_bar)} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def build_clipping_robustness(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    clip_ratios=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02),
    eval_snr: float | None = None,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = "docs/benchmarks",
) -> ClippingRobustnessResult:
    """Train one net, evaluate accuracy across a clip-severity sweep.

    For each ``clip_ratio`` a fresh balanced eval set is generated, each
    **waveform** hard-clipped to ``clip_ratio * peak`` then rescaled back to its
    original peak (level preserved, only flat-topping introduced), re-extracted
    to log-mel and scored. Imports inside the function keep torch/training off the
    pure reduction path.
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

    def _clip_wave(wave, clip_ratio):
        """Hard-clip to clip_ratio*peak, then rescale back to the original peak."""
        peak = wave.abs().max()
        if float(peak) == 0.0 or clip_ratio >= 1.0:
            return wave, 0.0
        thr = clip_ratio * peak
        clipped_mask = wave.abs() > thr
        clipped = wave.clamp(-thr, thr)
        clipped = clipped * (peak / thr)  # restore peak: isolates distortion from level
        return clipped, float(clipped_mask.float().mean())

    def _clipped_eval(clip_ratio: float, ds_seed: int):
        gen = torch.Generator().manual_seed(ds_seed)
        feats, labels, frac_sum = [], [], 0.0
        for stressed in (False, True):
            for _ in range(eval_n_per_class):
                wave = _synth_waveform(stressed, gen)
                if eval_snr is not None:
                    wave = _add_noise(wave, eval_snr, gen, color="white")
                wave, frac = _clip_wave(wave, clip_ratio)
                frac_sum += frac
                feats.append(extract(wave))
                labels.append(float(stressed))
        x = torch.cat(feats, dim=0)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return x, y, frac_sum / (2 * eval_n_per_class)

    records = []
    for clip_ratio in clip_ratios:
        x, y, clipped_frac = _clipped_eval(clip_ratio, seed + 7)
        with torch.no_grad():
            pred = (model(x).flatten() >= 0.5).float()
        acc = (pred == y.flatten()).float().mean().item()
        records.append((clip_ratio, acc, clipped_frac, x.shape[0]))

    return clipping_robustness(records, accuracy_bar=accuracy_bar, out_dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clipping / saturation robustness A/B")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_clipping_robustness(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/clipping_robustness.json and .md")


if __name__ == "__main__":
    main()
