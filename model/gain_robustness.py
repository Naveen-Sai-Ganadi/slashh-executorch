"""Input-gain (level) robustness A/B.

Every robustness artifact so far (``robustness``, ``int8_robustness``,
``noise_colors``, ``noise_failure_mode`` …) varies the **signal-to-noise ratio** —
additive noise at a fixed signal level. None varies the absolute input **level**.
That gap matters because the log-mel front-end (``model/features.py``) is
``log10(mel + eps)`` with **no per-window level normalization** — no mean
subtraction, no peak-norm. Scaling the captured waveform by a gain ``g`` scales
the mel power by ``g**2``, i.e. shifts every non-floored log-mel bin by a constant
``2*log10(g)``. So how loudly the user speaks, or the device's mic gain / AGC,
feeds the model a DC-shifted feature map it was never normalized against. A
1,549-param conv net may or may not have learned to tolerate that shift.

This A/B sweeps an input-gain range (dB), **re-extracts features from gained
waveforms** (so the eps-floor nonlinearity is faithful rather than a feature-
domain additive approximation), and measures accuracy at each level. It reports
the contiguous gain band around unity over which detection stays reliable — the
dynamic range the model tolerates before quiet or loud speech breaks it. Gain is
applied *after* noise, so the SNR is held constant and only the absolute level
moves: this isolates the level axis from the noise axis the other artifacts cover.

Host-only: CPU training + the pure-Python front-end, no device, no AI Hub token.
The reduction is pure (unit-tested with hand-set accuracies);
``build_gain_robustness`` trains one tiny net and evaluates gained eval sets.

Run:
    PYTHONPATH=. .venv/bin/python -m model.gain_robustness
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "GainPoint",
    "GainRobustnessResult",
    "gain_robustness",
    "build_gain_robustness",
]

# Project-wide "reliable" accuracy convention (matches robustness threshold).
_DEFAULT_ACCURACY_BAR = 0.8


@dataclass(frozen=True)
class GainPoint:
    """Accuracy at one input-gain level (``gain_db``; 0 dB = unity = unchanged)."""

    gain_db: float
    accuracy: float
    n: int

    @property
    def gain_factor(self) -> float:
        """Amplitude multiplier for this gain in dB (``10 ** (gain_db / 20)``)."""
        return 10.0 ** (self.gain_db / 20.0)

    def reliable(self, bar: float) -> bool:
        return self.accuracy >= bar

    def to_dict(self) -> dict:
        return {
            "gain_db": self.gain_db,
            "gain_factor": self.gain_factor,
            "accuracy": self.accuracy,
            "n": self.n,
        }


@dataclass(frozen=True)
class GainRobustnessResult:
    """Level-robustness reduction over a gain sweep."""

    points: list[GainPoint]  # sorted by gain_db ascending
    accuracy_bar: float

    @property
    def _unity(self) -> GainPoint | None:
        for p in self.points:
            if p.gain_db == 0.0:
                return p
        return None

    @property
    def unity_accuracy(self) -> float | None:
        u = self._unity
        return None if u is None else u.accuracy

    @property
    def worst_accuracy(self) -> float:
        return min(p.accuracy for p in self.points)

    @property
    def worst_gain_db(self) -> float:
        return min(self.points, key=lambda p: p.accuracy).gain_db

    @property
    def level_invariant(self) -> bool:
        """Every point in the sweep clears the bar."""
        return all(p.reliable(self.accuracy_bar) for p in self.points)

    def _band(self) -> tuple[float, float] | None:
        """Maximal contiguous reliable run (in gain order) containing unity."""
        u = self._unity
        if u is None or not u.reliable(self.accuracy_bar):
            return None
        idx = self.points.index(u)
        lo = idx
        while lo - 1 >= 0 and self.points[lo - 1].reliable(self.accuracy_bar):
            lo -= 1
        hi = idx
        while hi + 1 < len(self.points) and self.points[hi + 1].reliable(self.accuracy_bar):
            hi += 1
        return self.points[lo].gain_db, self.points[hi].gain_db

    @property
    def reliable_low_db(self) -> float | None:
        b = self._band()
        return None if b is None else b[0]

    @property
    def reliable_high_db(self) -> float | None:
        b = self._band()
        return None if b is None else b[1]

    @property
    def usable_band_db(self) -> float | None:
        b = self._band()
        return None if b is None else b[1] - b[0]

    @property
    def verdict(self) -> str:
        if self.level_invariant:
            return (
                "**Level-invariant**: detection clears the "
                f"{self.accuracy_bar:.0%} bar across the entire gain sweep "
                f"(worst {self.worst_accuracy:.0%} at {self.worst_gain_db:+g} dB). "
                "The model tolerates the un-normalized log-mel level shift — quiet "
                "or loud speech doesn't break it, so no input AGC is required."
            )
        b = self._band()
        if b is None:
            return (
                "**Fails at unity**: accuracy is below the "
                f"{self.accuracy_bar:.0%} bar even at 0 dB (unchanged level) — the "
                "level sweep can't be characterized around a broken operating "
                "point. Re-check the model before tuning gain."
            )
        return (
            "**Reliable within a band**: detection holds above the "
            f"{self.accuracy_bar:.0%} bar only for input gain in "
            f"[{b[0]:+g}, {b[1]:+g}] dB (a {b[1] - b[0]:g} dB window); outside it "
            f"accuracy falls to {self.worst_accuracy:.0%} at {self.worst_gain_db:+g} "
            "dB. Because the front-end isn't level-normalized, very quiet or loud "
            "capture shifts the log-mel map out of the learned range — add input "
            "AGC or a per-window level normalization to widen the usable range."
        )

    def to_dict(self) -> dict:
        return {
            "accuracy_bar": self.accuracy_bar,
            "n_points": len(self.points),
            "unity_accuracy": self.unity_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "worst_gain_db": self.worst_gain_db,
            "level_invariant": self.level_invariant,
            "reliable_low_db": self.reliable_low_db,
            "reliable_high_db": self.reliable_high_db,
            "usable_band_db": self.usable_band_db,
            "verdict": self.verdict,
            "points": [p.to_dict() for p in self.points],
        }


def gain_robustness(
    records,
    *,
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = None,
) -> GainRobustnessResult:
    """Reduce per-level ``(gain_db, accuracy, n)`` records.

    Points are sorted by ``gain_db``. Pure — no training, no device. Each accuracy
    must be in ``[0, 1]`` and ``n`` positive. Writes ``gain_robustness.{json,md}``
    to ``out_dir`` when given.
    """
    points = []
    for gain_db, acc, n in records:
        if not (0.0 <= float(acc) <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {acc}")
        if int(n) <= 0:
            raise ValueError("n must be positive")
        points.append(GainPoint(gain_db=float(gain_db), accuracy=float(acc), n=int(n)))
    if not points:
        raise ValueError("records must be non-empty")
    points.sort(key=lambda p: p.gain_db)

    out = GainRobustnessResult(points=points, accuracy_bar=accuracy_bar)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gain_robustness.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "gain_robustness.md").write_text(to_markdown(out))

    return out


def _fmt(x: float | None, unit: str = "") -> str:
    return "—" if x is None else f"{x:g}{unit}"


def to_markdown(out: GainRobustnessResult) -> str:
    band = out._band()
    band_txt = (
        f"[{band[0]:+g}, {band[1]:+g}] dB ({band[1] - band[0]:g} dB wide)"
        if band else "—"
    )
    header = (
        "# Input-gain (level) robustness (A/B)\n\n"
        "The log-mel front-end is `log10(mel + eps)` with no level normalization, "
        "so scaling the input waveform by gain `g` shifts every log-mel bin by "
        "`2*log10(g)`. This sweeps input gain (dB), re-extracts features from "
        "gained waveforms, and measures accuracy — the dynamic range the model "
        "tolerates before quiet/loud capture breaks detection. Gain is applied "
        "after noise, so SNR is held fixed.\n\n"
        f"- unity (0 dB) accuracy: **{_fmt(out.unity_accuracy)}**\n"
        f"- reliable gain band (bar {out.accuracy_bar:.0%}): **{band_txt}**\n"
        f"- level-invariant across sweep: **{out.level_invariant}**\n"
        f"- worst: **{out.worst_accuracy:.0%}** at **{out.worst_gain_db:+g} dB**\n"
        f"- {out.verdict}\n\n"
        "| gain (dB) | gain x | accuracy | reliable |\n"
        "|---|---|---|---|\n"
    )
    rows = [
        f"| {p.gain_db:+g} | {p.gain_factor:.3g} | {p.accuracy:.0%} "
        f"| {p.reliable(out.accuracy_bar)} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def build_gain_robustness(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    gain_db_levels=(-60.0, -48.0, -36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0),
    eval_snr: float | None = None,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = "docs/benchmarks",
) -> GainRobustnessResult:
    """Train one net, evaluate accuracy across an input-gain sweep.

    For each ``gain_db`` a fresh balanced eval set is generated, the **waveform**
    scaled by the gain factor (after optional noise at ``eval_snr``, so SNR is
    fixed), then re-extracted to log-mel and scored. Imports inside the function
    keep torch/training off the pure reduction path.
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

    def _gained_eval(gain_factor: float, ds_seed: int):
        gen = torch.Generator().manual_seed(ds_seed)
        feats, labels = [], []
        for stressed in (False, True):
            for _ in range(eval_n_per_class):
                wave = _synth_waveform(stressed, gen)
                if eval_snr is not None:
                    wave = _add_noise(wave, eval_snr, gen, color="white")
                wave = wave * gain_factor  # input gain: scales whole captured signal
                feats.append(extract(wave))
                labels.append(float(stressed))
        x = torch.cat(feats, dim=0)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return x, y

    records = []
    for gain_db in gain_db_levels:
        factor = 10.0 ** (gain_db / 20.0)
        x, y = _gained_eval(factor, seed + 7)
        with torch.no_grad():
            pred = (model(x).flatten() >= 0.5).float()
        acc = (pred == y.flatten()).float().mean().item()
        records.append((gain_db, acc, x.shape[0]))

    return gain_robustness(records, accuracy_bar=accuracy_bar, out_dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Input-gain (level) robustness A/B")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_gain_robustness(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/gain_robustness.json and .md")


if __name__ == "__main__":
    main()
