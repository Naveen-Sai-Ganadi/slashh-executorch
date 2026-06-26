"""Does *noise-aware* INT8 calibration preserve the robustness floor better?

The shipped deployable is the INT8 ``.pte`` (``quantize_production``). PT2E
static quantization observes activation ranges on a **calibration** batch to
choose the integer scales — and today that batch is *clean* synthetic audio
(``model/production.py``). But the production model trains noise-augmented (the
very-aggressive recipe ``{clean,20,10,5,0,-5,-10}``) and on-device it hears
*noisy* audio. If clean-only calibration under-observes the activation ranges the
model actually sees under noise, the quantizer could clip or coarsely bin those
activations and give back part of the noise-robustness floor.

This harness quantizes the **same** trained model two ways — calibrated on a
clean set vs. on a noise-augmented set drawn from the training SNR pool
(``robust_train.DEFAULT_AUG_SNRS``) — runs the identical waveform-noise SNR sweep
through both INT8 runtimes and the eager fp32 model (paired inputs via
``int8_robustness.PteModule``), and reports, per calibration, the reliable floor
and the per-SNR INT8−fp32 accuracy gap. The verdict names whether matching the
calibration distribution to deployment makes the floor **better**, the **same**,
or **worse** — and which calibration to ship.

Honest by design: for a net this small the predictions tend to stay put under
quantization, so "same" (clean calibration is already sufficient) is a real and
useful outcome, not a failure. Records ``docs/benchmarks/int8_calib_ab.{json,md}``.

    python -m model.int8_calib_ab            # train, quantize both ways, compare

Host-only: both INT8 programs run through the ExecuTorch host runtime, exactly
as on device. No AI Hub, no live token.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import synthetic_dataset
from .export_executorch import export_quantized_to_pte
from .int8_robustness import PteModule
from .model import StressNet
from .robust_train import DEFAULT_AUG_SNRS, noise_augmented_dataset
from .robustness import RobustnessResult, _snr_label, robustness_curve

__all__ = [
    "CalibResult",
    "CalibAbResult",
    "int8_calib_ab",
    "build_int8_calib_ab",
]


# None floor (reliable on clean only) sorts worst; deeper (more negative) is best.
def _floor_key(v: float | None) -> float:
    return float("inf") if v is None else v


@dataclass(frozen=True)
class CalibResult:
    """How one calibration's INT8 program did across the noise sweep."""

    label: str                              # "clean" | "noise-aware"
    calib_desc: str                         # the calibration distribution, in words
    int8: RobustnessResult
    reliable_floor_db: float | None         # deeper is better; None = clean-only
    acc_deltas: list[tuple[float | None, float]]   # (snr, int8_acc - fp32_acc)
    mean_abs_delta: float                   # mean |INT8 − fp32| accuracy over SNRs
    preserves_fp32_floor: bool              # INT8 floor at least as deep as fp32's

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "calib_desc": self.calib_desc,
            "reliable_floor_db": self.reliable_floor_db,
            "mean_abs_delta": self.mean_abs_delta,
            "preserves_fp32_floor": self.preserves_fp32_floor,
            "acc_deltas": [{"snr_db": s, "delta": d} for s, d in self.acc_deltas],
            "int8": self.int8.to_dict(),
        }


@dataclass(frozen=True)
class CalibAbResult:
    fp32: RobustnessResult
    fp32_reliable_floor_db: float | None
    results: list[CalibResult]              # [clean, noise-aware]
    threshold: float

    @property
    def clean(self) -> CalibResult:
        return self.results[0]

    @property
    def noise_aware(self) -> CalibResult:
        return self.results[1]

    @property
    def verdict(self) -> str:
        """Noise-aware floor vs clean floor: better / same / worse (deeper wins)."""
        nf = _floor_key(self.noise_aware.reliable_floor_db)
        cf = _floor_key(self.clean.reliable_floor_db)
        if nf < cf:
            return "better"
        if nf == cf:
            return "same"
        return "worse"

    @property
    def recommended(self) -> CalibResult:
        """Calibration with the deepest reliable floor; ties go to clean.

        Clean calibration is the cheaper, current default, so it only loses when
        noise-aware calibration *strictly* deepens the floor — we don't pay the
        extra calibration complexity for a tie.
        """
        if _floor_key(self.noise_aware.reliable_floor_db) < _floor_key(
            self.clean.reliable_floor_db
        ):
            return self.noise_aware
        return self.clean

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "fp32_reliable_floor_db": self.fp32_reliable_floor_db,
            "verdict": self.verdict,
            "recommended": self.recommended.label,
            "fp32": self.fp32.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }


def _evaluate_calibration(
    model: StressNet,
    fp32: RobustnessResult,
    calib_batch,
    *,
    label: str,
    calib_desc: str,
    snr_levels,
    n_per_class: int,
    seed: int,
    eval_seeds: Sequence[int] | None,
    threshold: float,
) -> CalibResult:
    """Quantize ``model`` with ``calib_batch``, sweep noise, diff against fp32."""
    pte = export_quantized_to_pte(model.eval(), calib_batch)
    int8 = robustness_curve(
        PteModule(pte), snr_levels=snr_levels, n_per_class=n_per_class,
        seed=seed, eval_seeds=eval_seeds, threshold=threshold,
    )
    deltas = [
        (q.snr_db, round(q.accuracy - f.accuracy, 4))
        for f, q in zip(fp32.points, int8.points)
    ]
    mean_abs = round(statistics.mean(abs(d) for _s, d in deltas), 4) if deltas else 0.0
    fp_floor, q_floor = fp32.reliable_floor_db, int8.reliable_floor_db
    preserves = (q_floor is None) == (fp_floor is None) and (
        fp_floor is None or (q_floor is not None and q_floor <= fp_floor)
    )
    return CalibResult(
        label=label,
        calib_desc=calib_desc,
        int8=int8,
        reliable_floor_db=q_floor,
        acc_deltas=deltas,
        mean_abs_delta=mean_abs,
        preserves_fp32_floor=preserves,
    )


def int8_calib_ab(
    model: StressNet,
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    calib_n: int = 24,
    calib_seed: int = 2024,
    aug_snrs: tuple[float | None, ...] = DEFAULT_AUG_SNRS,
    n_per_class: int = 64,
    seed: int = 1,
    eval_seeds: Sequence[int] | None = None,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> CalibAbResult:
    """A/B clean vs noise-aware INT8 calibration on one trained model.

    Both INT8 programs and the fp32 model are swept over the *same* seeded noisy
    inputs, so the only thing that differs between the two calibrations is the
    distribution PT2E observed when picking activation scales. ``calib_seed`` /
    ``calib_n`` size the calibration batches; the clean batch reproduces
    ``quantize_production``'s exactly so "clean" is the real shipped baseline.
    """
    model = model.eval()
    fp32 = robustness_curve(
        model, snr_levels=snr_levels, n_per_class=n_per_class, seed=seed,
        eval_seeds=eval_seeds, threshold=threshold,
    )

    clean_calib, _ = synthetic_dataset(calib_n, seed=calib_seed)
    noisy_calib, _ = noise_augmented_dataset(calib_n, seed=calib_seed, snr_choices=aug_snrs)

    clean = _evaluate_calibration(
        model, fp32, clean_calib, label="clean",
        calib_desc="clean synthetic audio (the current shipped calibration)",
        snr_levels=snr_levels, n_per_class=n_per_class, seed=seed,
        eval_seeds=eval_seeds, threshold=threshold,
    )
    noise_aware = _evaluate_calibration(
        model, fp32, noisy_calib, label="noise-aware",
        calib_desc=f"noise-augmented audio (SNR pool {_aug_label(aug_snrs)})",
        snr_levels=snr_levels, n_per_class=n_per_class, seed=seed,
        eval_seeds=eval_seeds, threshold=threshold,
    )

    out = CalibAbResult(
        fp32=fp32,
        fp32_reliable_floor_db=fp32.reliable_floor_db,
        results=[clean, noise_aware],
        threshold=threshold,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "int8_calib_ab.json").write_text(json.dumps(out.to_dict(), indent=2) + "\n")
        (out_dir / "int8_calib_ab.md").write_text(to_markdown(out))

    return out


def _aug_label(snrs: tuple[float | None, ...]) -> str:
    return "{" + ",".join("clean" if s is None else f"{s:g}" for s in snrs) + "}"


def _floor_label(snr_db: float | None) -> str:
    return "clean only" if snr_db is None else _snr_label(snr_db)


_VERDICT_GLOSS = {
    "better": (
        "**Noise-aware calibration deepens the floor** — matching the calibration "
        "distribution to deployment buys robustness the clean calibration left on "
        "the table. Ship the noise-aware calibration."
    ),
    "same": (
        "**No difference** — clean and noise-aware calibration hold the same "
        "reliable floor. At this size the quantizer's scales already cover the "
        "noisy activations, so the simpler clean calibration is sufficient; keep it."
    ),
    "worse": (
        "**Noise-aware calibration is worse** — calibrating on noisy audio widened "
        "the observed ranges and coarsened the scales on the clean/near-clean "
        "regime that matters most. Keep the clean calibration."
    ),
}


def to_markdown(out: CalibAbResult) -> str:
    rec = out.recommended
    verdict = (
        f"{_VERDICT_GLOSS[out.verdict]} Recommended calibration: "
        f"**`{rec.label}`** (reliable to {_floor_label(rec.reliable_floor_db)})."
    )
    header = (
        "# INT8 calibration A/B — clean vs noise-aware\n\n"
        "The shipped deployable is the INT8 `.pte`; PT2E picks its integer scales "
        "from a **calibration** batch. The same trained model is quantized two "
        "ways — calibrated on clean audio (today's default) vs on noise-augmented "
        "audio from the training SNR pool — then both INT8 runtimes and the eager "
        "**fp32** model are swept over *identical* noisy inputs, so the only "
        "variable is the calibration distribution.\n\n"
        f"- fp32 reliable down to: **{_floor_label(out.fp32_reliable_floor_db)}** "
        f"(accuracy ≥ {out.threshold:.2f})\n"
        f"- {verdict}\n\n"
        "| calibration | reliable floor | mean |Δ| vs fp32 | preserves fp32 floor |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for r in out.results:
        star = " ⭐" if r is rec else ""
        rows.append(
            f"| `{r.label}`{star} | {_floor_label(r.reliable_floor_db)} "
            f"| {r.mean_abs_delta:.3f} | {'yes' if r.preserves_fp32_floor else 'no'} |"
        )
    # Per-SNR fp32 vs each INT8 accuracy, for the full picture.
    grid = (
        "\n\n### Per-SNR accuracy\n\n"
        "| SNR | fp32 | INT8 (clean calib) | INT8 (noise-aware calib) |\n"
        "|---|---|---|---|\n"
    )
    clean_by = {p.snr_db: p.accuracy for p in out.clean.int8.points}
    na_by = {p.snr_db: p.accuracy for p in out.noise_aware.int8.points}
    grid_rows = [
        f"| {_snr_label(f.snr_db)} | {f.accuracy:.3f} | {clean_by[f.snr_db]:.3f} "
        f"| {na_by[f.snr_db]:.3f} |"
        for f in out.fp32.points
    ]
    return header + "\n".join(rows) + grid + "\n".join(grid_rows) + "\n"


def build_int8_calib_ab(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    eval_n_per_class: int = 64,
    calib_n: int = 24,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> CalibAbResult:
    """Train the production model and A/B its INT8 calibration distributions."""
    # Imported here to avoid a module-load cycle (production imports robustness).
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=threshold,
    )
    return int8_calib_ab(
        model, snr_levels=snr_levels, calib_n=calib_n,
        n_per_class=eval_n_per_class, seed=seed + 1, threshold=threshold,
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 clean vs noise-aware calibration A/B")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    ap.add_argument("--calib-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_int8_calib_ab(
        epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, calib_n=args.calib_n,
        seed=args.seed, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/int8_calib_ab.json and int8_calib_ab.md")


if __name__ == "__main__":
    main()
