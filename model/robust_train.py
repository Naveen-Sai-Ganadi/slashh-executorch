"""Noise-augmented training — raise the model's operating floor.

model/robustness.py showed the clean-trained StressNet is brittle: perfect on
clean audio but collapsing toward chance by ~10 dB SNR. The fix is to let the
model *see* noise during training. This module builds a noise-augmented dataset
(each window kept clean or corrupted at a randomly chosen SNR), trains on it,
and compares the resulting robustness curve against a clean-trained baseline so
the improvement is measured, not assumed.

    python -m model.robust_train          # train both, write the comparison

Augmentation is the cheapest robustness lever and runs entirely on the host —
no device, no AI Hub, no live token. The ``.pte`` export path is unchanged; a
robustly-trained checkpoint drops straight into export_to_pte.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import _synth_waveform
from .features import extract
from .model import StressNet
from .robustness import RobustnessResult, robustness_curve
from .train import train

# SNRs (dB) sampled per-window during augmentation; None = leave clean.
DEFAULT_AUG_SNRS: tuple[float | None, ...] = (None, 20.0, 10.0, 5.0)


def _add_noise(wave: torch.Tensor, snr_db: float, gen: torch.Generator) -> torch.Tensor:
    sig_power = wave.pow(2).mean().clamp(min=1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    return wave + torch.randn(wave.shape, generator=gen) * noise_power.sqrt()


def noise_augmented_dataset(
    n_per_class: int,
    seed: int = 0,
    snr_choices: tuple[float | None, ...] = DEFAULT_AUG_SNRS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced synthetic set where each window is clean or noised at a random SNR.

    Mirrors ``synthetic_dataset`` but, per window, draws an SNR from
    ``snr_choices`` and corrupts the waveform before feature extraction.
    Deterministic for a fixed ``seed``.
    """
    gen = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for stressed in (False, True):
        for _ in range(n_per_class):
            wave = _synth_waveform(stressed, gen)
            idx = int(torch.randint(0, len(snr_choices), (1,), generator=gen))
            snr = snr_choices[idx]
            if snr is not None:
                wave = _add_noise(wave, snr, gen)
            feats.append(extract(wave))
            labels.append(float(stressed))
    x = torch.cat(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    perm = torch.randperm(x.shape[0], generator=gen)
    return x[perm], y[perm]


@dataclass(frozen=True)
class RobustTrainResult:
    baseline: RobustnessResult     # clean-trained model's robustness curve
    augmented: RobustnessResult    # noise-augmented model's robustness curve

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "augmented": self.augmented.to_dict(),
        }


def compare_robustness(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 96,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 5.0, 0.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> RobustTrainResult:
    """Train clean vs noise-augmented; return both robustness curves.

    The two models share width, epochs, and optimizer settings — only the
    training data differs — so the curve gap is attributable to augmentation.
    """
    # baseline: clean synthetic training
    base_model = StressNet()
    train(data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
          n_per_class=n_per_class, seed=seed, model=base_model)

    # augmented: same recipe, noise-augmented data
    aug_model = StressNet()
    aug_data = noise_augmented_dataset(n_per_class, seed=seed)
    train(data_dir=None, epochs=epochs, batch_size=batch_size, lr=lr,
          n_per_class=0, seed=seed, model=aug_model, dataset=aug_data)

    eval_seed = seed + 1
    base_curve = robustness_curve(
        base_model, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=eval_seed, threshold=threshold,
    )
    aug_curve = robustness_curve(
        aug_model, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=eval_seed, threshold=threshold,
    )
    out = RobustTrainResult(baseline=base_curve, augmented=aug_curve)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "robust_train.json").write_text(json.dumps(out.to_dict(), indent=2) + "\n")
        (out_dir / "robust_train.md").write_text(to_markdown(out, threshold=threshold))

    return out


def _snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def _floor_label(floor_db: float | None) -> str:
    return "none (holds)" if floor_db is None else _snr_label(floor_db)


def to_markdown(out: RobustTrainResult, *, threshold: float = 0.8) -> str:
    """Side-by-side accuracy-vs-SNR for clean-trained vs noise-augmented."""
    base = {p.snr_db: p for p in out.baseline.points}
    header = (
        "# Noise-augmented training vs clean training\n\n"
        "Same architecture, epochs, and optimizer — only the training data "
        "differs. Accuracy is on noise-injected eval sets at each SNR. The "
        f"**operating floor** is the first SNR where accuracy < {threshold:.2f}.\n\n"
        f"- baseline (clean-trained) floor: **{_floor_label(out.baseline.floor_db)}**\n"
        f"- augmented floor: **{_floor_label(out.augmented.floor_db)}**\n\n"
        "| SNR | baseline acc | augmented acc | Δ |\n"
        "|---|---|---|---|\n"
    )
    rows = []
    for p in out.augmented.points:
        b = base.get(p.snr_db)
        b_acc = b.accuracy if b else float("nan")
        rows.append(
            f"| {_snr_label(p.snr_db)} | {b_acc:.3f} | {p.accuracy:.3f} | "
            f"{p.accuracy - b_acc:+.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Noise-augmented vs clean training")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    out = compare_robustness(
        epochs=args.epochs, n_per_class=args.n_per_class, seed=args.seed,
        threshold=args.threshold, out_dir=args.out_dir,
    )
    print(to_markdown(out, threshold=args.threshold))
    print(f"wrote {args.out_dir}/robust_train.json and robust_train.md")


if __name__ == "__main__":
    main()
