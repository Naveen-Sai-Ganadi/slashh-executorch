"""Reverberation robustness A/B.

The robustness suite already isolates *additive* noise (``robustness``),
*nonlinear* clipping (``clipping_robustness``), and *linear* level
(``gain_robustness``). The remaining common real-world distortion is
**convolutive**: room reverb. Hands-free / speakerphone / across-the-room capture
convolves the voice with a room impulse response — late reflections plus a decay
tail that smear the temporal envelope. That smearing is invisible to the additive
SNR, nonlinear clip, and linear-gain models the suite already owns, so it gets its
own axis here.

For each ``rt60_s`` the waveform is convolved with a synthetic exponential-decay
RIR (white noise enveloped to hit −60 dB at the RT60, direct path at tap 0) and
then **rescaled back to the original RMS**, so gross level/energy is unchanged and
only the convolutive smearing remains — reverb is separated cleanly from the level
axis ``gain_robustness`` already owns. It re-extracts log-mel features from the
reverberated waveforms and measures accuracy, reporting the most reverberant room
the model tolerates before the smeared envelope breaks detection.

``rt60_s`` = reverberation time in seconds (0.0 = dry / anechoic reference; ~0.3 a
small room, ~0.6 a live room, ~1.0 a hall). ``tail_fraction`` = the share of
output energy in the reverberant tail beyond the direct path, reported for
context.

Host-only: CPU training + the pure-Python front-end, no device, no AI Hub token.
The reduction is pure (unit-tested with hand-set accuracies);
``build_reverb_robustness`` trains one tiny net and evaluates reverberated eval
sets.

Run:
    PYTHONPATH=. .venv/bin/python -m model.reverb_robustness
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ReverbPoint",
    "ReverbRobustnessResult",
    "reverb_robustness",
    "build_reverb_robustness",
]

# Project-wide "reliable" accuracy convention (matches robustness threshold).
_DEFAULT_ACCURACY_BAR = 0.8

_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ReverbPoint:
    """Accuracy at one reverberation severity.

    ``rt60_s`` is the reverberation time in seconds (0.0 = dry reference).
    ``tail_fraction`` is the share of output energy in the reverberant tail
    (beyond the direct path), informational.
    """

    rt60_s: float
    accuracy: float
    tail_fraction: float
    n: int

    def reliable(self, bar: float) -> bool:
        return self.accuracy >= bar

    def to_dict(self) -> dict:
        return {
            "rt60_s": self.rt60_s,
            "accuracy": self.accuracy,
            "tail_fraction": self.tail_fraction,
            "n": self.n,
        }


@dataclass(frozen=True)
class ReverbRobustnessResult:
    """Reverberation-robustness reduction over an RT60 sweep."""

    points: list[ReverbPoint]  # sorted by rt60_s ascending (dry first)
    accuracy_bar: float

    @property
    def _dry(self) -> ReverbPoint | None:
        for p in self.points:
            if p.rt60_s == 0.0:
                return p
        return None

    @property
    def clean_accuracy(self) -> float | None:
        d = self._dry
        return None if d is None else d.accuracy

    @property
    def worst_accuracy(self) -> float:
        return min(p.accuracy for p in self.points)

    @property
    def worst_rt60_s(self) -> float:
        return max(self.points, key=lambda p: (-p.accuracy, p.rt60_s)).rt60_s

    @property
    def reverb_tolerant(self) -> bool:
        """Every point in the sweep clears the bar."""
        return all(p.reliable(self.accuracy_bar) for p in self.points)

    @property
    def reliable_ceiling_s(self) -> float | None:
        """Highest ``rt60_s`` still reliable, contiguous from the dry (0.0) end.

        Points are ordered dry→reverberant (rt60_s ascending). Walk up from the
        anechoic reference while each successive point clears the bar; the last
        such RT60 is the most reverberant room tolerated. ``None`` if there is no
        dry reference or it already fails the bar.
        """
        d = self._dry
        if d is None or not d.reliable(self.accuracy_bar):
            return None
        ceiling = d.rt60_s
        idx = self.points.index(d)
        i = idx
        while i + 1 < len(self.points) and self.points[i + 1].reliable(self.accuracy_bar):
            i += 1
            ceiling = self.points[i].rt60_s
        return ceiling

    @property
    def verdict(self) -> str:
        if self.reverb_tolerant:
            return (
                "**Reverb-tolerant**: detection clears the "
                f"{self.accuracy_bar:.0%} bar across the entire RT60 sweep "
                f"(worst {self.worst_accuracy:.0%} at RT60 {self.worst_rt60_s:g}s). "
                "Room reflections and a decay tail from hands-free / across-the-room "
                "capture don't break detection, so no dereverberation is required."
            )
        if self._dry is None or not self._dry.reliable(self.accuracy_bar):
            return (
                "**Fails dry**: accuracy is below the "
                f"{self.accuracy_bar:.0%} bar even at RT60 0.0s (anechoic) — the "
                "reverb sweep can't be characterized around a broken operating "
                "point. Re-check the model before tuning reverb tolerance."
            )
        ceiling = self.reliable_ceiling_s
        return (
            "**Reliable up to a reverb ceiling**: detection holds above the "
            f"{self.accuracy_bar:.0%} bar only while RT60 stays at or below "
            f"**{ceiling:g}s**; beyond that, envelope smearing drops accuracy to "
            f"{self.worst_accuracy:.0%} at RT60 {self.worst_rt60_s:g}s. Reverberant "
            "rooms or distant / hands-free capture past this will be missed — bring "
            "the mic closer, add dereverberation, or train with reverberant "
            "augmentation to widen tolerance."
        )

    def to_dict(self) -> dict:
        return {
            "accuracy_bar": self.accuracy_bar,
            "n_points": len(self.points),
            "clean_accuracy": self.clean_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "worst_rt60_s": self.worst_rt60_s,
            "reverb_tolerant": self.reverb_tolerant,
            "reliable_ceiling_s": self.reliable_ceiling_s,
            "verdict": self.verdict,
            "points": [p.to_dict() for p in self.points],
        }


def reverb_robustness(
    records,
    *,
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = None,
) -> ReverbRobustnessResult:
    """Reduce per-severity ``(rt60_s, accuracy, tail_fraction, n)`` records.

    Points are sorted by ``rt60_s`` ascending (dry first). Pure — no training,
    no device. ``rt60_s`` must be ``>= 0`` (0.0 = dry reference), ``accuracy``
    and ``tail_fraction`` in ``[0, 1]``, and ``n`` positive. Writes
    ``reverb_robustness.{json,md}`` to ``out_dir`` when given.
    """
    points = []
    for rt60_s, acc, tail_frac, n in records:
        if float(rt60_s) < 0.0:
            raise ValueError(f"rt60_s must be >= 0, got {rt60_s}")
        if not (0.0 <= float(acc) <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {acc}")
        if not (0.0 <= float(tail_frac) <= 1.0):
            raise ValueError(f"tail_fraction must be in [0, 1], got {tail_frac}")
        if int(n) <= 0:
            raise ValueError("n must be positive")
        points.append(ReverbPoint(
            rt60_s=float(rt60_s), accuracy=float(acc),
            tail_fraction=float(tail_frac), n=int(n),
        ))
    if not points:
        raise ValueError("records must be non-empty")
    points.sort(key=lambda p: p.rt60_s)

    out = ReverbRobustnessResult(points=points, accuracy_bar=accuracy_bar)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reverb_robustness.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "reverb_robustness.md").write_text(to_markdown(out))

    return out


def _fmt(x: float | None, unit: str = "") -> str:
    return "—" if x is None else f"{x:g}{unit}"


def to_markdown(out: ReverbRobustnessResult) -> str:
    ceiling = out.reliable_ceiling_s
    ceiling_txt = f"{ceiling:g}s" if ceiling is not None else "—"
    header = (
        "# Reverberation robustness (A/B)\n\n"
        "The suite already covers additive noise, nonlinear clipping, and linear "
        "gain; this isolates the *convolutive* distortion a real room produces: "
        "reverb. Each waveform is convolved with a synthetic exponential-decay RIR "
        "(−60 dB at the RT60, direct path at tap 0), then rescaled back to the "
        "original RMS (level/energy held fixed, only temporal smearing remains), "
        "re-extracted to log-mel, and scored — the most reverberant room the model "
        "tolerates before the smeared envelope breaks detection.\n\n"
        f"- dry (RT60 0.0s) accuracy: **{_fmt(out.clean_accuracy)}**\n"
        f"- reliable reverb ceiling (bar {out.accuracy_bar:.0%}): **{ceiling_txt}**\n"
        f"- reverb-tolerant across sweep: **{out.reverb_tolerant}**\n"
        f"- worst: **{out.worst_accuracy:.0%}** at **RT60 {out.worst_rt60_s:g}s**\n"
        f"- {out.verdict}\n\n"
        "| rt60 (s) | reverb tail energy | accuracy | reliable |\n"
        "|---|---|---|---|\n"
    )
    rows = [
        f"| {p.rt60_s:g} | {p.tail_fraction:.0%} | {p.accuracy:.0%} "
        f"| {p.reliable(out.accuracy_bar)} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def _reverb_wave(wave, rt60_s: float, gen, sr: int = _SAMPLE_RATE):
    """Convolve with an exponential-decay RIR, then rescale back to original RMS.

    Returns ``(reverbed_wave, tail_fraction)``. ``tail_fraction`` is the share of
    pre-rescale output energy beyond the direct path. RT60 0 (or non-positive) is
    the dry passthrough. Isolates the convolutive smearing from gross level.
    """
    import torch

    if rt60_s <= 0.0:
        return wave, 0.0
    flat = wave.flatten()
    n = flat.shape[0]
    rms_in = flat.pow(2).mean().sqrt()
    if float(rms_in) == 0.0:
        return wave, 0.0

    length = max(2, int(rt60_s * sr))
    t = torch.arange(length, dtype=torch.float32)
    env = torch.pow(torch.tensor(10.0), -3.0 * t / float(length))  # -60 dB at RT60
    noise = torch.rand(length, generator=gen) * 2.0 - 1.0
    rir = noise * env
    rir[0] = 1.0  # direct path

    # linear convolution via FFT, keep the first n (causal) samples
    nfft = 1
    while nfft < n + length - 1:
        nfft *= 2
    y = torch.fft.irfft(torch.fft.rfft(flat, nfft) * torch.fft.rfft(rir, nfft), nfft)[:n]

    total_e = y.pow(2).sum()
    direct_e = (flat * rir[0]).pow(2).sum()
    tail_frac = float((1.0 - (direct_e / total_e)).clamp(0.0, 1.0)) if float(total_e) > 0 else 0.0

    rms_out = y.pow(2).mean().sqrt()
    if float(rms_out) > 0:
        y = y * (rms_in / rms_out)  # restore RMS: isolates smearing from level/energy
    return y.reshape(wave.shape), tail_frac


def build_reverb_robustness(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    rt60s=(0.0, 0.15, 0.3, 0.6, 1.0),
    eval_snr: float | None = None,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = "docs/benchmarks",
) -> ReverbRobustnessResult:
    """Train one net, evaluate accuracy across an RT60 sweep.

    For each ``rt60_s`` a fresh balanced eval set is generated, each **waveform**
    convolved with a synthetic exponential-decay RIR then rescaled back to its
    original RMS (level preserved, only convolutive smearing introduced),
    re-extracted to log-mel and scored. Imports inside the function keep
    torch/training off the pure reduction path.
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

    def _reverbed_eval(rt60_s: float, ds_seed: int):
        gen = torch.Generator().manual_seed(ds_seed)
        rir_gen = torch.Generator().manual_seed(ds_seed + 101)
        feats, labels, frac_sum = [], [], 0.0
        for stressed in (False, True):
            for _ in range(eval_n_per_class):
                wave = _synth_waveform(stressed, gen)
                if eval_snr is not None:
                    wave = _add_noise(wave, eval_snr, gen, color="white")
                wave, frac = _reverb_wave(wave, rt60_s, rir_gen)
                frac_sum += frac
                feats.append(extract(wave))
                labels.append(float(stressed))
        x = torch.cat(feats, dim=0)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return x, y, frac_sum / (2 * eval_n_per_class)

    records = []
    for rt60_s in rt60s:
        x, y, tail_frac = _reverbed_eval(rt60_s, seed + 7)
        with torch.no_grad():
            pred = (model(x).flatten() >= 0.5).float()
        acc = (pred == y.flatten()).float().mean().item()
        records.append((rt60_s, acc, tail_frac, x.shape[0]))

    return reverb_robustness(records, accuracy_bar=accuracy_bar, out_dir=out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reverberation robustness A/B")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_reverb_robustness(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/reverb_robustness.json and .md")


if __name__ == "__main__":
    main()
