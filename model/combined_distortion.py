"""Combined (simultaneous) field-distortion robustness (A/B).

Every prior robustness item isolates ONE axis — additive noise (``robustness``),
nonlinear clipping (``clipping_robustness``), convolutive reverb
(``reverb_robustness``), linear level (``gain_robustness``). Real capture stacks
all of them at once. This composes them in the physical signal-chain order a mic
actually sees — **gain -> reverb -> noise -> clip** — and quantifies the
*compounding gap*: how far simultaneous distortion drops accuracy below what the
single-axis sweeps would predict.

Rationale for the order: input level (gain) is set at the mic, the room convolves
it (reverb), ambient noise adds at the mic sum (noise), then the ADC/preamp
hard-clips (clip). That is the realistic chain and it is applied in that sequence.

Like its siblings the module is two layers: a **pure reduction**
(``combined_distortion``) unit-tested with hand-set numbers, and a heavy
``build_combined_distortion`` that trains one tiny net and evaluates field-
distorted eval sets. The injector ``apply_field_profile`` reuses the proven
single-axis distortions: it imports ``_add_noise`` from ``model.robustness``,
replicates ``clipping_robustness``'s ``_clip_wave`` (hard-clip to ``ratio*peak``
then rescale to the original peak), and applies the same exp-decay-RIR + RMS-
restore reverb as ``reverb_robustness._reverb_wave`` with one physical refinement:
the diffuse tail enters ~20 dB below the direct path (a realistic direct-to-
reflection ratio) instead of at full level. The sibling's full-level diffuse tail
carries far more energy than the direct path (an unrealistic, reverb-dominated
RIR); keeping the direct path dominant means a harsher profile genuinely perturbs
the waveform more than a milder one, so the field severities stay monotone. A
clean/identity profile is a true no-op: it returns the waveform bit-for-bit and
draws no RNG, so the clean profile reproduces the undistorted baseline exactly.

For each profile the build also measures the **matched single-axis accuracy** —
the min accuracy over the profile's *active* axes evaluated one at a time with the
same model and the same eval seeds — so the reported compounding gap is data-
backed, not asserted.

Host-only: CPU training + the pure-Python front-end, no device, no AI Hub token.
The reduction path imports no torch.

Run:
    PYTHONPATH=. .venv/bin/python -m model.combined_distortion
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FieldProfile",
    "DEFAULT_PROFILES",
    "apply_field_profile",
    "ProfilePoint",
    "CombinedDistortionResult",
    "combined_distortion",
    "to_markdown",
    "build_combined_distortion",
]

# Project-wide "reliable" accuracy convention (matches robustness threshold).
_DEFAULT_ACCURACY_BAR = 0.8

_SAMPLE_RATE = 16000

# Diffuse-tail level relative to the direct path (-20 dB). Real early reflections
# sit ~10-20 dB under the direct sound; this keeps the direct path dominant so
# harsher field profiles perturb the waveform more than milder ones (monotone
# severity). The sibling reverb uses a full-level (reverb-dominated) tail.
_REFLECTION_LEVEL = 0.1


def _round(x: float | None, ndigits: int = 6) -> float | None:
    """Round for byte-stable JSON; pass ``None`` through unchanged."""
    return None if x is None else round(float(x), ndigits)


@dataclass(frozen=True)
class FieldProfile:
    """A named simultaneous-distortion field condition.

    The four physical axes are applied in mic-chain order on ``apply_field_profile``:
    ``gain_db`` (input level, factor ``10**(gain_db/20)``, applied FIRST),
    ``rt60_s`` (reverberation time in seconds; ``0.0`` = dry; applied SECOND),
    ``snr_db`` (additive-noise SNR in dB; ``None`` = no noise; applied THIRD), and
    ``clip_ratio`` (hard-clip threshold as a fraction of peak; ``>=1.0`` = no clip;
    applied LAST). ``noise_color`` selects the noise spectrum.
    """

    name: str
    gain_db: float
    rt60_s: float
    snr_db: float | None
    clip_ratio: float
    noise_color: str = "white"

    def is_clean(self) -> bool:
        """True only when every axis is degenerate (a true identity profile)."""
        return (
            self.gain_db == 0.0
            and self.rt60_s == 0.0
            and self.snr_db is None
            and self.clip_ratio >= 1.0
        )


# Named field profiles. Severities are drawn from the already-shipped single-axis
# sweeps so they are directly matchable to those benchmarks.
DEFAULT_PROFILES: tuple[FieldProfile, ...] = (
    FieldProfile("clean", 0.0, 0.0, None, 1.0),          # degenerate identity
    FieldProfile("quiet", -3.0, 0.15, 20.0, 0.9),        # small quiet office
    FieldProfile("typical", -6.0, 0.30, 10.0, 0.6),      # normal room + background
    FieldProfile("harsh", -12.0, 0.60, 0.0, 0.35),       # loud reverberant, clipping mic
    FieldProfile("worst_case", -12.0, 1.0, -5.0, 0.25),  # corner: ~worst of each axis
)


# --- injector: compose the physical signal chain ----------------------------


def _reverb_wave(wave, rt60_s: float, gen, sr: int = _SAMPLE_RATE):
    """Convolve with an exp-decay RIR, then rescale back to original RMS.

    Same shape as ``reverb_robustness._reverb_wave``: a synthetic exponential-decay
    room impulse response (-60 dB at the RT60, direct path at tap 0), linear
    convolution kept causal to ``n`` samples, then RMS-restored so only the
    convolutive smearing remains. The one refinement vs the sibling: the diffuse
    tail enters at ``_REFLECTION_LEVEL`` (-20 dB) below the direct path so the
    direct path stays dominant (a realistic direct-to-reflection ratio), which
    keeps harsher RT60s genuinely more perturbing. Returns
    ``(reverbed_wave, tail_fraction)``; RT60 <= 0 is the dry passthrough (no RNG
    draw).
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
    rir = noise * env * _REFLECTION_LEVEL  # diffuse tail ~20 dB below direct
    rir[0] = 1.0  # direct path dominates

    nfft = 1
    while nfft < n + length - 1:
        nfft *= 2
    y = torch.fft.irfft(torch.fft.rfft(flat, nfft) * torch.fft.rfft(rir, nfft), nfft)[:n]

    total_e = y.pow(2).sum()
    direct_e = (flat * rir[0]).pow(2).sum()
    tail_frac = (
        float((1.0 - (direct_e / total_e)).clamp(0.0, 1.0)) if float(total_e) > 0 else 0.0
    )

    rms_out = y.pow(2).mean().sqrt()
    if float(rms_out) > 0:
        y = y * (rms_in / rms_out)  # restore RMS: isolate smearing from level/energy
    return y.reshape(wave.shape), tail_frac


def _clip_wave(wave, clip_ratio: float):
    """Hard-clip to ``clip_ratio*peak``, then rescale back to the original peak.

    Replicates the ``_clip_wave`` in ``clipping_robustness.build_clipping_robustness``.
    Returns ``(clipped_wave, clipped_fraction)``; ``clip_ratio >= 1`` (or a silent
    waveform) is the identity passthrough.
    """
    peak = wave.abs().max()
    if float(peak) == 0.0 or clip_ratio >= 1.0:
        return wave, 0.0
    thr = clip_ratio * peak
    clipped_mask = wave.abs() > thr
    clipped = wave.clamp(-thr, thr)
    clipped = clipped * (peak / thr)  # restore peak: isolate distortion from level
    return clipped, float(clipped_mask.float().mean())


def apply_field_profile(wave, profile: FieldProfile, gen):
    """Compose gain -> reverb -> noise -> clip for one field profile.

    Returns ``(distorted_wave, diagnostics)`` where ``diagnostics`` carries
    ``tail_fraction`` (reverberant tail energy) and ``clipped_fraction`` (share of
    clipped samples), both ``0.0`` for stages that are inactive. A clean/identity
    profile (``profile.is_clean()``) is guarded *before* any generator use, so it
    returns the waveform bit-for-bit and draws no RNG — the clean profile then
    reproduces the undistorted features exactly.
    """
    if profile.is_clean():
        return wave, {"tail_fraction": 0.0, "clipped_fraction": 0.0}

    from .robustness import _add_noise

    out = wave
    # 1) gain (linear level; deterministic, no RNG)
    if profile.gain_db != 0.0:
        out = out * (10.0 ** (profile.gain_db / 20.0))
    # 2) reverb (convolutive)
    tail_fraction = 0.0
    if profile.rt60_s > 0.0:
        out, tail_fraction = _reverb_wave(out, profile.rt60_s, gen)
    # 3) noise (additive)
    if profile.snr_db is not None:
        out = _add_noise(out, profile.snr_db, gen, color=profile.noise_color)
    # 4) clip (nonlinear; LAST, as the ADC/preamp sees the summed signal)
    clipped_fraction = 0.0
    if profile.clip_ratio < 1.0:
        out, clipped_fraction = _clip_wave(out, profile.clip_ratio)

    return out, {"tail_fraction": tail_fraction, "clipped_fraction": clipped_fraction}


# --- pure reduction ---------------------------------------------------------


@dataclass(frozen=True)
class ProfilePoint:
    """Accuracy at one field profile (the four axes pin the severity).

    ``matched_single_axis_acc`` is the min accuracy over the profile's active axes
    measured one at a time; ``compounding_gap`` = matched - combined (>= 0 means
    simultaneous distortion hurts more than the single-axis prediction).
    """

    name: str
    gain_db: float
    rt60_s: float
    snr_db: float | None
    clip_ratio: float
    accuracy: float
    acc_std: float | None
    n: int
    matched_single_axis_acc: float | None = None

    def is_clean(self) -> bool:
        return (
            self.gain_db == 0.0
            and self.rt60_s == 0.0
            and self.snr_db is None
            and self.clip_ratio >= 1.0
        )

    def reliable(self, bar: float) -> bool:
        return self.accuracy >= bar

    def compounding_gap(self) -> float | None:
        if self.matched_single_axis_acc is None:
            return None
        return self.matched_single_axis_acc - self.accuracy

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gain_db": self.gain_db,
            "rt60_s": self.rt60_s,
            "snr_db": self.snr_db,
            "clip_ratio": self.clip_ratio,
            "accuracy": _round(self.accuracy),
            "acc_std": _round(self.acc_std),
            "n": self.n,
            "matched_single_axis_acc": _round(self.matched_single_axis_acc),
            "compounding_gap": _round(self.compounding_gap()),
        }


@dataclass(frozen=True)
class CombinedDistortionResult:
    """Combined-distortion reduction over a set of field profiles."""

    points: tuple[ProfilePoint, ...]  # input order (clean -> worst by convention)
    accuracy_bar: float

    def _clean_point(self) -> ProfilePoint | None:
        for p in self.points:
            if p.is_clean():
                return p
        return None

    def clean_accuracy(self) -> float | None:
        c = self._clean_point()
        return None if c is None else c.accuracy

    def worst_accuracy(self) -> float:
        return min(p.accuracy for p in self.points)

    def reliable_profiles(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.points if p.reliable(self.accuracy_bar))

    def max_compounding_gap(self) -> float | None:
        gaps = [
            p.compounding_gap()
            for p in self.points
            if not p.is_clean() and p.compounding_gap() is not None
        ]
        return max(gaps) if gaps else None

    def graceful(self) -> bool:
        """Every profile (including the worst case) clears the bar."""
        return all(p.reliable(self.accuracy_bar) for p in self.points)

    def anomalous_profiles(self, tol: float = 0.03) -> tuple[str, ...]:
        """Non-clean profiles that scored ABOVE their matched single-axis
        baseline (compounding gap < -tol).

        A negative compounding gap is not physically meaningful as genuine
        robustness: stacking *more* distortion on top of a single axis cannot
        truly raise accuracy. When it happens it flags that the synthetic eval
        under-stresses that particular combination, so a "graceful" verdict that
        rests on such a point is bounded by the synthetic eval rather than proof
        of field robustness.
        """
        names: list[str] = []
        for p in self.points:
            if p.is_clean():
                continue
            g = p.compounding_gap()
            if g is not None and g < -tol:
                names.append(p.name)
        return tuple(names)

    def _last_reliable_name(self) -> str | None:
        """Last profile in the contiguous reliable run from the clean end."""
        last: str | None = None
        for p in self.points:
            if p.reliable(self.accuracy_bar):
                last = p.name
            else:
                break
        return last

    def verdict(self) -> str:
        bar = self.accuracy_bar
        if self.graceful():
            base = (
                "**Graceful composition**: robustness holds under simultaneous "
                f"distortion — every field profile clears the {bar:.0%} bar (worst "
                f"{self.worst_accuracy():.0%}). Single-axis robustness is not "
                "overstated, so no combined-distortion augmentation is required."
            )
        else:
            clean = self.clean_accuracy()
            if clean is not None and clean >= bar:
                last = self._last_reliable_name() or "—"
                gap = self.max_compounding_gap()
                gap_txt = "—" if gap is None else f"{gap:.3f}"
                base = (
                    "**Compounding degradation**: single-axis robustness OVERSTATES "
                    f"field robustness — accuracy clears the {bar:.0%} bar clean but "
                    f"drops to {self.worst_accuracy():.0%} under simultaneous "
                    f"distortion; reliable only through **{last}**; max compounding "
                    f"gap {gap_txt}. Fix: combined-distortion augmentation "
                    "(augmentation_ab / recipe_envelope)."
                )
            else:
                base = (
                    f"**Fails clean**: accuracy is below the {bar:.0%} bar even on "
                    "the clean identity profile — the field-distortion sweep can't "
                    "be characterized around a broken operating point. Re-check the "
                    "model first."
                )
        anomalies = self.anomalous_profiles()
        if anomalies:
            gaps = {p.name: p.compounding_gap() for p in self.points}
            worst = min(anomalies, key=lambda nm: gaps[nm])
            base += (
                f" Caveat: {', '.join(anomalies)} scored ABOVE the matched "
                f"single-axis baseline (gap {gaps[worst]:+.3f}) — stacking more "
                "distortion cannot genuinely raise accuracy, so the synthetic eval "
                "under-stresses that combination; read the verdict as bounded by "
                "synthetic-eval limits, not proof of field robustness."
            )
        return base

    def to_dict(self) -> dict:
        return {
            "accuracy_bar": self.accuracy_bar,
            "n_points": len(self.points),
            "clean_accuracy": _round(self.clean_accuracy()),
            "worst_accuracy": _round(self.worst_accuracy()),
            "graceful": self.graceful(),
            "max_compounding_gap": _round(self.max_compounding_gap()),
            "reliable_profiles": list(self.reliable_profiles()),
            "anomalous_profiles": list(self.anomalous_profiles()),
            "verdict": self.verdict(),
            "points": [p.to_dict() for p in self.points],
        }


def combined_distortion(
    records,
    *,
    single_axis: dict | None = None,
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = "docs/benchmarks",
) -> CombinedDistortionResult:
    """Reduce per-profile records into a :class:`CombinedDistortionResult`.

    ``records`` is a list of tuples
    ``(name, gain_db, rt60_s, snr_db, clip_ratio, accuracy, acc_std, n)``. The
    optional ``single_axis`` maps a profile name to its matched single-axis
    accuracy (min over the profile's active axes). Input order is preserved
    (clean -> worst by convention). Pure — no torch, no device. Accuracy must be
    in ``[0, 1]`` and ``n`` positive. Writes ``combined_distortion.{json,md}`` to
    ``out_dir`` when given.
    """
    single_axis = single_axis or {}
    points: list[ProfilePoint] = []
    for name, gain_db, rt60_s, snr_db, clip_ratio, acc, acc_std, n in records:
        if not (0.0 <= float(acc) <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {acc}")
        if int(n) <= 0:
            raise ValueError("n must be positive")
        points.append(
            ProfilePoint(
                name=str(name),
                gain_db=float(gain_db),
                rt60_s=float(rt60_s),
                snr_db=None if snr_db is None else float(snr_db),
                clip_ratio=float(clip_ratio),
                accuracy=float(acc),
                acc_std=None if acc_std is None else float(acc_std),
                n=int(n),
                matched_single_axis_acc=(
                    None if single_axis.get(name) is None else float(single_axis[name])
                ),
            )
        )
    if not points:
        raise ValueError("records must be non-empty")

    out = CombinedDistortionResult(points=tuple(points), accuracy_bar=accuracy_bar)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "combined_distortion.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "combined_distortion.md").write_text(to_markdown(out))

    return out


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _fmt_gap(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def to_markdown(out: CombinedDistortionResult) -> str:
    header = (
        "# Combined multi-distortion field robustness (A/B)\n\n"
        "Every prior robustness item isolates ONE axis (additive noise, nonlinear "
        "clipping, convolutive reverb, linear gain); real capture stacks them at "
        "once. This composes them in the physical signal-chain order a mic sees — "
        "**gain -> reverb -> noise -> clip** — re-extracts log-mel, and scores. It "
        "reports the *compounding gap*: how far simultaneous distortion drops "
        "accuracy below the matched single-axis prediction.\n\n"
        f"- clean (identity) accuracy: **{_fmt_pct(out.clean_accuracy())}**\n"
        f"- worst profile accuracy: **{out.worst_accuracy():.0%}**\n"
        f"- graceful across all profiles (bar {out.accuracy_bar:.0%}): "
        f"**{out.graceful()}**\n"
        f"- max compounding gap: **{_fmt_gap(out.max_compounding_gap())}**\n"
        f"- reliable profiles: {', '.join(out.reliable_profiles()) or '—'}\n"
        f"- {out.verdict()}\n\n"
        "| profile | gain (dB) | rt60 (s) | snr (dB) | clip_ratio | accuracy "
        "| single-axis | compounding gap | reliable |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for p in out.points:
        snr_txt = "—" if p.snr_db is None else f"{p.snr_db:g}"
        single_txt = (
            "—"
            if p.matched_single_axis_acc is None
            else f"{p.matched_single_axis_acc:.0%}"
        )
        gap = p.compounding_gap()
        gap_txt = "—" if gap is None else f"{gap:+.3f}"
        std_txt = "" if p.acc_std is None else f" ± {p.acc_std:.3f}"
        rows.append(
            f"| {p.name} | {p.gain_db:+g} | {p.rt60_s:g} | {snr_txt} "
            f"| {p.clip_ratio:g} | {p.accuracy:.0%}{std_txt} | {single_txt} "
            f"| {gap_txt} | {p.reliable(out.accuracy_bar)} |"
        )
    note = (
        "\n\n_Methodology & caveats: the reverb stage convolves a synthetic "
        "exponential-decay RIR whose diffuse tail enters ~20 dB below the direct "
        "path (a realistic direct-to-reflection ratio), so the reverb axis is "
        "intentionally milder than a fully reverb-dominated room — the harsh and "
        "worst_case profiles therefore understate extreme reverberation. A "
        "negative compounding gap means a profile scored above its matched "
        "single-axis baseline, which is not physically meaningful (stacking more "
        "distortion cannot raise accuracy) and flags that the synthetic eval "
        "under-stresses that combination rather than demonstrating real "
        "robustness._\n"
    )
    return header + "\n".join(rows) + note


# --- heavy build path -------------------------------------------------------


def build_combined_distortion(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    profiles: tuple[FieldProfile, ...] = DEFAULT_PROFILES,
    eval_seeds=(0, 1, 2),
    accuracy_bar: float = _DEFAULT_ACCURACY_BAR,
    out_dir: str | Path | None = "docs/benchmarks",
) -> CombinedDistortionResult:
    """Train one net, evaluate accuracy across simultaneous-distortion profiles.

    For each profile a balanced eval set is generated per ``eval_seed`` with one
    ``torch.Generator().manual_seed(eval_seed)`` driving both the synthetic
    waveforms and the distortion draws, then ``apply_field_profile`` composes
    gain -> reverb -> noise -> clip, features are re-extracted, and accuracy is
    averaged (mean +/- std) across the eval seeds. The matched single-axis
    accuracy is the min over the profile's active axes evaluated one at a time
    with the same model and the same eval seeds, so the compounding gap is data-
    backed. Imports inside the function keep torch off the pure reduction path.
    """
    import statistics

    import torch

    from .data import _synth_waveform
    from .features import extract
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed, threshold=0.8
    )
    model = model.eval()

    def _eval_profile(profile: FieldProfile):
        accs: list[float] = []
        n_items = 0
        for eval_seed in eval_seeds:
            gen = torch.Generator().manual_seed(eval_seed)
            feats, labels = [], []
            for stressed in (False, True):
                for _ in range(eval_n_per_class):
                    wave = _synth_waveform(stressed, gen)
                    dwave, _diag = apply_field_profile(wave, profile, gen)
                    feats.append(extract(dwave))
                    labels.append(float(stressed))
            x = torch.cat(feats, dim=0)
            y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                pred = (model(x).flatten() >= 0.5).float()
            accs.append((pred == y.flatten()).float().mean().item())
            n_items = x.shape[0]
        mean_acc = statistics.fmean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else None
        return mean_acc, std_acc, n_items * len(list(eval_seeds))

    def _matched_single_axis(profile: FieldProfile) -> float | None:
        axes: list[FieldProfile] = []
        if profile.gain_db != 0.0:
            axes.append(
                FieldProfile(f"{profile.name}/gain", profile.gain_db, 0.0, None, 1.0)
            )
        if profile.rt60_s > 0.0:
            axes.append(
                FieldProfile(f"{profile.name}/reverb", 0.0, profile.rt60_s, None, 1.0)
            )
        if profile.snr_db is not None:
            axes.append(
                FieldProfile(
                    f"{profile.name}/noise",
                    0.0,
                    0.0,
                    profile.snr_db,
                    1.0,
                    profile.noise_color,
                )
            )
        if profile.clip_ratio < 1.0:
            axes.append(
                FieldProfile(f"{profile.name}/clip", 0.0, 0.0, None, profile.clip_ratio)
            )
        if not axes:
            return None
        return min(_eval_profile(ax)[0] for ax in axes)

    records = []
    single_axis: dict[str, float] = {}
    for profile in profiles:
        mean_acc, std_acc, total_n = _eval_profile(profile)
        records.append(
            (
                profile.name,
                profile.gain_db,
                profile.rt60_s,
                profile.snr_db,
                profile.clip_ratio,
                mean_acc,
                std_acc,
                total_n,
            )
        )
        matched = _matched_single_axis(profile)
        if matched is not None:
            single_axis[profile.name] = matched

    return combined_distortion(
        records, single_axis=single_axis, accuracy_bar=accuracy_bar, out_dir=out_dir
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined multi-distortion field robustness A/B"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_combined_distortion(
        seed=args.seed,
        epochs=args.epochs,
        n_per_class=args.n_per_class,
        eval_seeds=(0, 1, 2),
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/combined_distortion.json and .md")


if __name__ == "__main__":
    main()
