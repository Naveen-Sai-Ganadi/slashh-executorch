"""Per-channel vs per-tensor INT8 granularity A/B.

The shipped INT8 ``.pte`` quantizes weights *per channel* (``is_per_channel=
True`` in ``export_quantized_to_pte``): a separate scale/zero-point for each
output channel of every conv/linear. The simpler alternative is *per-tensor* —
one scale/zero-point for the whole weight tensor. Per-tensor is smaller (no
per-channel scale vectors in the program) and is sometimes the only mode a
fixed-function NPU backend supports, but it is coarser: a layer whose channels
have very different weight magnitudes loses resolution.

Every other INT8 artifact here holds granularity fixed at per-channel and
varies something else — calibration data (int8_calib_ab), noise
(int8_robustness), drift (int8_calibration_drift), file size (pte_footprint).
This A/B asks the granularity question itself: on the 1,549-param production
net, does per-tensor give back accuracy or an SNR step of the robustness floor,
and how much smaller is the program?

Honest angle: a tiny net's weights are unlikely to vary enough channel-to-
channel for per-tensor to hurt, so the likely finding is "per-tensor is the
cheaper equivalent". That is a real result worth documenting, not dressing up.

Host-only: both INT8 programs run through the ExecuTorch host runtime exactly
as on device (``int8_robustness.PteModule``). No AI Hub, no live token. The
reduction is pure (unit-tested without exporting); ``build_int8_granularity_ab``
trains one net and exports it twice.

Run:
    PYTHONPATH=. .venv/bin/python -m model.int8_granularity_ab
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "GranularityPoint",
    "GranularityResult",
    "int8_granularity_ab",
    "build_int8_granularity_ab",
]

# An accuracy gap this large (at any SNR) between the two granularities is
# "material" — big enough that the granularity choice is a real accuracy lever,
# not quantization/sampling noise.
_MATERIAL_ACC_GAP = 0.05


def _reliable_floor(pairs, *, threshold):
    """Lowest numeric SNR (scanning clean->noisy) still at/above ``threshold``.

    Mirrors :func:`model.robustness.reliable_floor` on ``(snr_db, accuracy)``
    pairs: the clean point (``snr_db is None``) is never itself a floor, and the
    scan stops at the first sub-threshold row (contiguous from clean).
    """
    floor = None
    for snr, acc in pairs:
        if acc < threshold:
            break
        if snr is not None:
            floor = snr
    return floor


@dataclass(frozen=True)
class GranularityPoint:
    """Per-channel vs per-tensor accuracy at one SNR (``snr_db=None`` = clean)."""

    snr_db: float | None
    acc_per_channel: float
    acc_per_tensor: float

    @property
    def delta(self) -> float:
        """per-tensor minus per-channel (positive = per-tensor is better here)."""
        return self.acc_per_tensor - self.acc_per_channel

    def to_dict(self) -> dict:
        d = asdict(self)
        d["delta"] = self.delta
        return d


@dataclass(frozen=True)
class GranularityResult:
    """Granularity A/B over an SNR sweep plus the two programs' byte sizes."""

    points: list[GranularityPoint]
    threshold: float
    per_channel_bytes: int
    per_tensor_bytes: int

    @property
    def floor_per_channel(self) -> float | None:
        return _reliable_floor(
            [(p.snr_db, p.acc_per_channel) for p in self.points],
            threshold=self.threshold,
        )

    @property
    def floor_per_tensor(self) -> float | None:
        return _reliable_floor(
            [(p.snr_db, p.acc_per_tensor) for p in self.points],
            threshold=self.threshold,
        )

    @property
    def per_tensor_holds_floor(self) -> bool:
        """True if per-tensor is reliable down to at least as low an SNR.

        Same None semantics as ``int8_robustness.preserves_floor``: a missing
        per-channel floor (reliable clean-only) is held only if per-tensor is
        also clean-only.
        """
        c, t = self.floor_per_channel, self.floor_per_tensor
        if c is None:
            return t is None
        return t is not None and t <= c

    @property
    def max_abs_delta(self) -> float:
        """Largest absolute per-SNR accuracy gap between the two granularities."""
        return max(abs(p.delta) for p in self.points)

    @property
    def material_accuracy_gap(self) -> bool:
        return self.max_abs_delta >= _MATERIAL_ACC_GAP

    @property
    def byte_savings_pct(self) -> float:
        """Percent of the per-channel ``.pte`` shed by going per-tensor."""
        return (1.0 - self.per_tensor_bytes / self.per_channel_bytes) * 100.0

    @property
    def worth_per_channel(self) -> bool:
        """True if per-channel earns its cost — per-tensor either gives back an
        SNR step of the floor or diverges materially somewhere on the curve."""
        return not (self.per_tensor_holds_floor and not self.material_accuracy_gap)

    @property
    def verdict(self) -> str:
        save = (
            f"per-tensor is {self.byte_savings_pct:+.1f}% on file size "
            f"({self.per_channel_bytes}->{self.per_tensor_bytes} bytes)"
        )
        if not self.per_tensor_holds_floor:
            return (
                "**Per-channel earns its keep**: per-tensor gives back the "
                f"robustness floor (per-channel reliable to {self.floor_per_channel}, "
                f"per-tensor only to {self.floor_per_tensor}). Keep per-channel — "
                f"{save}, but the coarser scale costs an SNR step."
            )
        if self.material_accuracy_gap:
            return (
                "**Mixed**: per-tensor matches the floor but diverges by up to "
                f"{self.max_abs_delta:.3f} accuracy mid-curve. Per-channel is the "
                f"safer default; {save} is not worth the wobble."
            )
        return (
            "**Per-tensor is the cheaper equivalent**: it holds the same "
            f"reliable floor ({self.floor_per_tensor}) with no material accuracy "
            f"gap (max |Δ| {self.max_abs_delta:.3f}). For this tiny net the "
            f"per-channel scale vectors buy nothing — {save}."
        )

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "per_channel_bytes": self.per_channel_bytes,
            "per_tensor_bytes": self.per_tensor_bytes,
            "byte_savings_pct": self.byte_savings_pct,
            "floor_per_channel": self.floor_per_channel,
            "floor_per_tensor": self.floor_per_tensor,
            "per_tensor_holds_floor": self.per_tensor_holds_floor,
            "max_abs_delta": self.max_abs_delta,
            "material_accuracy_gap": self.material_accuracy_gap,
            "worth_per_channel": self.worth_per_channel,
            "verdict": self.verdict,
            "points": [p.to_dict() for p in self.points],
        }


def int8_granularity_ab(
    points,
    *,
    threshold: float = 0.8,
    per_channel_bytes: int,
    per_tensor_bytes: int,
    out_dir: str | Path | None = None,
) -> GranularityResult:
    """Reduce an SNR sweep of (per-channel, per-tensor) accuracies to a verdict.

    ``points`` is an iterable of ``(snr_db, acc_per_channel, acc_per_tensor)``
    triples, clean->noisy. Pure — no export, no device. Both byte counts must be
    positive (the program sizes). Writes ``int8_granularity_ab.{json,md}`` to
    ``out_dir`` when given.
    """
    pts = [
        GranularityPoint(snr_db=s, acc_per_channel=float(c), acc_per_tensor=float(t))
        for s, c, t in points
    ]
    if not pts:
        raise ValueError("points must be non-empty")
    if per_channel_bytes <= 0 or per_tensor_bytes <= 0:
        raise ValueError("per_channel_bytes and per_tensor_bytes must be positive")

    out = GranularityResult(
        points=pts,
        threshold=threshold,
        per_channel_bytes=int(per_channel_bytes),
        per_tensor_bytes=int(per_tensor_bytes),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "int8_granularity_ab.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "int8_granularity_ab.md").write_text(to_markdown(out))

    return out


def _snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def to_markdown(out: GranularityResult) -> str:
    header = (
        "# Per-channel vs per-tensor INT8 (A/B)\n\n"
        "The same trained production net, quantized two ways: the shipped "
        "**per-channel** scheme (a scale/zero-point per output channel) vs "
        "**per-tensor** (one scale for the whole tensor — smaller, coarser, and "
        "sometimes the only mode a fixed-function NPU supports). Both INT8 "
        "programs run through the ExecuTorch host runtime on *identical* noisy "
        "inputs, so each row is paired.\n\n"
        f"- per-channel reliable to: **{_snr_label(out.floor_per_channel)}** "
        f"(accuracy ≥ {out.threshold:.2f})\n"
        f"- per-tensor reliable to: **{_snr_label(out.floor_per_tensor)}**\n"
        f"- program size: per-channel **{out.per_channel_bytes:,}** B · "
        f"per-tensor **{out.per_tensor_bytes:,}** B "
        f"(**{out.byte_savings_pct:+.1f}%**)\n"
        f"- max |Δ accuracy|: **{out.max_abs_delta:.3f}** "
        f"(material ≥ {_MATERIAL_ACC_GAP:g}: **{out.material_accuracy_gap}**)\n"
        f"- {out.verdict}\n\n"
        "| SNR | per-channel acc | per-tensor acc | Δ (PT−PC) |\n"
        "|---|---|---|---|\n"
    )
    rows = [
        f"| {_snr_label(p.snr_db)} | {p.acc_per_channel:.3f} | "
        f"{p.acc_per_tensor:.3f} | {p.delta:+.3f} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def build_int8_granularity_ab(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    eval_n_per_class: int = 64,
    calib_n: int = 24,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    threshold: float = 0.8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> GranularityResult:
    """Train one production net, export INT8 per-channel + per-tensor, compare.

    Both programs are calibrated on the *same* clean synthetic batch, so the
    only difference is weight granularity. Imports inside the function to keep
    the export/quant stack off the pure reduction import path.
    """
    from .data import synthetic_dataset
    from .export_executorch import export_quantized_to_pte
    from .int8_robustness import PteModule
    from .production import train_production
    from .robustness import robustness_curve

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=threshold,
    )
    model.eval()

    calib, _ = synthetic_dataset(calib_n, seed=seed + 2024)
    per_channel_pte = export_quantized_to_pte(model, calib, per_channel=True)
    per_tensor_pte = export_quantized_to_pte(model, calib, per_channel=False)

    pc_curve = robustness_curve(
        PteModule(per_channel_pte), snr_levels=snr_levels,
        n_per_class=eval_n_per_class, seed=seed + 1, threshold=threshold,
    )
    pt_curve = robustness_curve(
        PteModule(per_tensor_pte), snr_levels=snr_levels,
        n_per_class=eval_n_per_class, seed=seed + 1, threshold=threshold,
    )
    by_snr = {p.snr_db: p for p in pt_curve.points}
    points = [
        (pc.snr_db, pc.accuracy, by_snr[pc.snr_db].accuracy)
        for pc in pc_curve.points
    ]

    return int8_granularity_ab(
        points,
        threshold=threshold,
        per_channel_bytes=len(per_channel_pte),
        per_tensor_bytes=len(per_tensor_pte),
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-channel vs per-tensor INT8 granularity A/B"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--eval-n-per-class", type=int, default=64)
    args = ap.parse_args()
    out = build_int8_granularity_ab(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        eval_n_per_class=args.eval_n_per_class, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/int8_granularity_ab.json and .md")


if __name__ == "__main__":
    main()
