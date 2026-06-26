"""Does the robustness floor hold under realistic (colored) noise?

``model/robustness.py`` sweeps *white* Gaussian noise — a worst-case flat
spectrum. Real acoustic noise the on-device classifier actually hears (traffic,
fans, HVAC, room rumble) is colored: pink (``1/f`` power) and brown (``1/f^2``)
concentrate energy in the low frequencies, exactly where the log-mel features
carry most of their weight. This harness re-runs the same SNR sweep for each
noise color and reports the reliable floor per color, so the robustness claim is
not an artifact of one convenient noise model.

    python -m model.noise_colors            # train, sweep white/pink/brown, record

Records ``docs/benchmarks/noise_colors.{json,md}``. Host-only; no device, no
AI Hub token.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .model import StressNet
from .robustness import RobustnessResult, _snr_label, robustness_curve

__all__ = [
    "NoiseColorResult",
    "noise_color_robustness",
    "build_noise_color_robustness",
    "COLORS",
]

COLORS = ("white", "pink", "brown")


@dataclass(frozen=True)
class NoiseColorResult:
    curves: dict[str, RobustnessResult]
    threshold: float

    @property
    def reliable_floor_db(self) -> dict[str, float | None]:
        return {c: r.reliable_floor_db for c, r in self.curves.items()}

    @property
    def holds_across_colors(self) -> bool:
        """True if every color is reliable down to at least one *noisy* SNR."""
        return all(f is not None for f in self.reliable_floor_db.values())

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "reliable_floor_db": self.reliable_floor_db,
            "holds_across_colors": self.holds_across_colors,
            "curves": {c: r.to_dict() for c, r in self.curves.items()},
        }


def noise_color_robustness(
    model: StressNet,
    *,
    colors: Sequence[str] = COLORS,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    n_per_class: int = 64,
    seed: int = 1,
    eval_seeds: Sequence[int] | None = None,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> NoiseColorResult:
    """Run the noise sweep once per spectral color; record per-color floors."""
    model = model.eval()
    curves = {
        color: robustness_curve(
            model, snr_levels=snr_levels, n_per_class=n_per_class, seed=seed,
            eval_seeds=eval_seeds, noise_color=color, threshold=threshold,
        )
        for color in colors
    }
    out = NoiseColorResult(curves=curves, threshold=threshold)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "noise_colors.json").write_text(json.dumps(out.to_dict(), indent=2) + "\n")
        (out_dir / "noise_colors.md").write_text(to_markdown(out))

    return out


def _floor_label(snr_db: float | None) -> str:
    return "clean only" if snr_db is None else _snr_label(snr_db)


def to_markdown(out: NoiseColorResult) -> str:
    verdict = (
        "**The floor holds across colors** — the model stays reliable into noisy "
        "SNRs for white, pink, and brown noise alike."
        if out.holds_across_colors
        else "**At least one color breaks the floor** — the model is reliable only "
        "on clean audio for some noise color; lean on smoothing in that regime."
    )
    floors = out.reliable_floor_db
    header = (
        "# Robustness across noise colors\n\n"
        "The same waveform-noise SNR sweep is run for each spectral color. White "
        "is flat; **pink** (`1/f`) and **brown** (`1/f²`) put more energy in the "
        "low frequencies, closer to real traffic/fan/HVAC noise.\n\n"
        + "".join(
            f"- {color} reliable down to: **{_floor_label(floors[color])}**\n"
            for color in out.curves
        )
        + f"- {verdict}\n\n"
    )
    # one accuracy column per color, shared SNR rows
    color_list = list(out.curves)
    head = "| SNR | " + " | ".join(f"{c} acc" for c in color_list) + " |\n"
    sep = "|---|" + "|".join("---" for _ in color_list) + "|\n"
    # SNR grid is identical across colors; take it from the first curve
    snrs = [p.snr_db for p in out.curves[color_list[0]].points]
    acc = {c: {p.snr_db: p.accuracy for p in out.curves[c].points} for c in color_list}
    rows = [
        f"| {_snr_label(s)} | " + " | ".join(f"{acc[c][s]:.3f}" for c in color_list) + " |"
        for s in snrs
    ]
    return header + head + sep + "\n".join(rows) + "\n"


def build_noise_color_robustness(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    eval_n_per_class: int = 64,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> NoiseColorResult:
    """Train the production model and sweep robustness across noise colors."""
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=threshold,
    )
    return noise_color_robustness(
        model, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=seed + 1, threshold=threshold, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustness across noise colors")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_noise_color_robustness(
        epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, seed=args.seed, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/noise_colors.json and noise_colors.md")


if __name__ == "__main__":
    main()
