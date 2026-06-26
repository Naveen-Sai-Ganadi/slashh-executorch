"""Does INT8 quantization preserve the noise-robustness floor?

The shipped deployable is the INT8 ``.pte`` (``quantize_production``), but its
fidelity was only ever checked on *clean* audio (``_int8_max_abs_diff`` — scores
within ~0.004 of eager). That is necessary but not sufficient: on-device the
classifier hears noisy audio, and quantization error can compound with acoustic
noise. The project advertises a robustness floor measured on the *fp32* model;
this module checks whether the INT8 runtime actually holds that floor.

It runs the same waveform-noise sweep (``model/robustness.py``) through both the
eager fp32 model and the INT8 ``.pte`` — on *identical* noisy inputs, so the
comparison is paired — and reports the per-SNR accuracy gap plus a one-line
verdict: does INT8 preserve the reliable floor, or does it give one SNR step
back? Records the evidence under ``docs/benchmarks/int8_robustness.{json,md}``.

    python -m model.int8_robustness            # train, quantize, compare, record

Host-only: the INT8 program runs through the ExecuTorch host runtime
(``run_pte``), exactly as on device. No AI Hub, no live token.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from executorch.runtime import Runtime

from .model import StressNet
from .robustness import RobustnessResult, _snr_label, robustness_curve

__all__ = [
    "PteModule",
    "Int8RobustnessResult",
    "int8_robustness",
    "build_int8_robustness",
]


class PteModule:
    """Adapt a ``.pte`` to the model interface ``robustness_curve`` expects.

    ``robustness_curve`` only needs ``.eval()`` and ``model(x) -> [N, 1]``. The
    exported program has a fixed batch-1 input, so a batched call is fanned out
    one sample at a time through the loaded method — the same thing
    ``run_pte`` does per call, but the program is loaded once and reused.
    """

    def __init__(self, pte_bytes: bytes) -> None:
        # Keep the temp file alive for the lifetime of the loaded program.
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "int8.pte"
        path.write_bytes(pte_bytes)
        program = Runtime.get().load_program(path)
        self._method = program.load_method("forward")

    def eval(self) -> PteModule:
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self._method.execute([x[i : i + 1]])[0] for i in range(x.shape[0])]
        return torch.cat(outs, dim=0)


@dataclass(frozen=True)
class Int8RobustnessResult:
    fp32: RobustnessResult
    int8: RobustnessResult
    # (snr_db, int8_accuracy - fp32_accuracy) per SNR, clean->noisy.
    acc_deltas: list[tuple[float | None, float]]
    threshold: float

    @property
    def preserves_floor(self) -> bool:
        """True if INT8 is reliable down to at least as low an SNR as fp32.

        Lower SNR = more noise, so a numerically lower (or equal) reliable floor
        means INT8 gave nothing back. A missing fp32 floor (reliable clean-only)
        is preserved only if INT8 is also clean-only.
        """
        f, q = self.fp32.reliable_floor_db, self.int8.reliable_floor_db
        if f is None:
            return q is None
        return q is not None and q <= f

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "fp32_reliable_floor_db": self.fp32.reliable_floor_db,
            "int8_reliable_floor_db": self.int8.reliable_floor_db,
            "preserves_floor": self.preserves_floor,
            "acc_deltas": [
                {"snr_db": snr, "delta": d} for snr, d in self.acc_deltas
            ],
            "fp32": self.fp32.to_dict(),
            "int8": self.int8.to_dict(),
        }


def int8_robustness(
    model: StressNet,
    int8_pte: bytes,
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    n_per_class: int = 64,
    seed: int = 1,
    eval_seeds: Sequence[int] | None = None,
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
) -> Int8RobustnessResult:
    """Sweep noise through both the fp32 model and its INT8 ``.pte``; compare.

    Both curves are evaluated with the same seeds, so each SNR feeds *identical*
    noisy inputs to both — the per-SNR accuracy delta is a paired measurement of
    quantization's effect under noise, not a sampling artifact.
    """
    model = model.eval()
    fp32 = robustness_curve(
        model, snr_levels=snr_levels, n_per_class=n_per_class, seed=seed,
        eval_seeds=eval_seeds, threshold=threshold,
    )
    int8 = robustness_curve(
        PteModule(int8_pte), snr_levels=snr_levels, n_per_class=n_per_class,
        seed=seed, eval_seeds=eval_seeds, threshold=threshold,
    )
    deltas = [
        (q.snr_db, round(q.accuracy - f.accuracy, 4))
        for f, q in zip(fp32.points, int8.points)
    ]
    out = Int8RobustnessResult(fp32=fp32, int8=int8, acc_deltas=deltas, threshold=threshold)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "int8_robustness.json").write_text(json.dumps(out.to_dict(), indent=2) + "\n")
        (out_dir / "int8_robustness.md").write_text(to_markdown(out))

    return out


def _floor_label(snr_db: float | None) -> str:
    return "clean only" if snr_db is None else _snr_label(snr_db)


def to_markdown(out: Int8RobustnessResult) -> str:
    verdict = (
        "**INT8 preserves the floor** — the deployable is reliable down to at "
        "least the same SNR as the fp32 model."
        if out.preserves_floor
        else "**INT8 degrades the floor** — quantization gives back at least one "
        "SNR step versus the fp32 model; lean on smoothing near the boundary."
    )
    header = (
        "# INT8 vs fp32 noise robustness\n\n"
        "The shipped deployable is the INT8 `.pte`. The same waveform-noise "
        "sweep is run through both the eager **fp32** model and the **INT8** "
        "ExecuTorch runtime on *identical* inputs, so each row is a paired "
        "comparison.\n\n"
        f"- fp32 reliable down to: **{_floor_label(out.fp32.reliable_floor_db)}** "
        f"(accuracy ≥ {out.threshold:.2f})\n"
        f"- INT8 reliable down to: **{_floor_label(out.int8.reliable_floor_db)}**\n"
        f"- {verdict}\n\n"
        "| SNR | fp32 acc | INT8 acc | Δ (INT8−fp32) |\n"
        "|---|---|---|---|\n"
    )
    by_snr = {p.snr_db: p for p in out.int8.points}
    rows = []
    for f in out.fp32.points:
        q = by_snr[f.snr_db]
        rows.append(
            f"| {_snr_label(f.snr_db)} | {f.accuracy:.3f} | {q.accuracy:.3f} | "
            f"{q.accuracy - f.accuracy:+.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def build_int8_robustness(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    eval_n_per_class: int = 64,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> Int8RobustnessResult:
    """Train the production model, quantize it, and compare INT8↔fp32 robustness."""
    # Imported here to avoid a module-load cycle (production imports robustness).
    from .production import quantize_production, train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=threshold,
    )
    pte = quantize_production(model)
    return int8_robustness(
        model, pte, snr_levels=snr_levels, n_per_class=eval_n_per_class,
        seed=seed + 1, threshold=threshold, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 vs fp32 robustness comparison")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_int8_robustness(
        epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, seed=args.seed, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/int8_robustness.json and int8_robustness.md")


if __name__ == "__main__":
    main()
