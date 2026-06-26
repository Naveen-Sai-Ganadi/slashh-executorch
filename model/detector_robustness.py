"""Does the *detector* (not just the model) survive noise — and how fast?

``model/robustness.py`` and ``model/noise_failure_mode.py`` score the raw model
one window at a time. The product, though, ships ``StressDetector``: EMA
smoothing + dual-threshold hysteresis over a *stream* of windows
(model/detector.py, mirrored on-device in ``StressPipeline.kt``). Smoothing
trades latency for stability — it can hold a decision steady where a single
noisy window would wobble, but it cannot recover information the model has
already lost. So the end-to-end question is its own measurement:

* **detection rate** — on stressed traces, does the latch reach "stressed"?
* **false-alarm rate** — on calm traces, does it ever latch by mistake?
* **latency** — how many windows until it first latches (the cost smoothing
  buys with).

This harness streams multi-window noisy traces through ``HostStressPipeline``
(a fresh detector per trace) and reports all three per SNR, plus the lowest SNR
that still meets a detect/false-alarm target — the *detection* floor, the
end-to-end analogue of the raw model's reliable floor.

    python -m model.detector_robustness

Records ``docs/benchmarks/detector_robustness.{json,md}``. Host-only; no device,
no AI Hub token.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import _synth_waveform
from .detector import StressDetector
from .model import StressNet
from .pipeline import HostStressPipeline
from .robustness import _add_noise, _snr_label

__all__ = [
    "DetectorSNRPoint",
    "DetectorRobustnessResult",
    "detector_robustness",
    "build_detector_robustness",
]


@dataclass(frozen=True)
class DetectorSNRPoint:
    """End-to-end detector outcome at one SNR (``snr_db=None`` = clean)."""

    snr_db: float | None
    detect_rate: float          # stressed traces that latch "stressed"
    false_alarm_rate: float     # calm traces that latch by mistake
    median_latency_windows: float | None  # windows-to-latch over detected traces
    n_traces: int               # per class

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DetectorRobustnessResult:
    points: list[DetectorSNRPoint]
    window_count: int
    detect_target: float
    fa_tolerance: float
    # lowest noisy SNR still meeting detect_rate >= target AND false_alarm_rate
    # <= tolerance — the end-to-end "detection floor". None if no noisy SNR
    # qualifies (clean-only, or never).
    detection_floor_db: float | None

    def to_dict(self) -> dict:
        return {
            "window_count": self.window_count,
            "detect_target": self.detect_target,
            "fa_tolerance": self.fa_tolerance,
            "detection_floor_db": self.detection_floor_db,
            "points": [p.to_dict() for p in self.points],
        }


def _run_trace(
    pipe: HostStressPipeline, stressed: bool, snr: float | None,
    window_count: int, gen: torch.Generator,
) -> int | None:
    """Stream one trace; return the 1-based window index where it first latched.

    Returns ``None`` if the latch never reached "stressed" within the trace.
    """
    pipe.reset()
    latched_at: int | None = None
    for w in range(1, window_count + 1):
        wave = _synth_waveform(stressed, gen)
        if snr is not None:
            wave = _add_noise(wave, snr, gen)
        state = pipe.process_window(wave, voiced=True)
        if state.stressed and latched_at is None:
            latched_at = w
    return latched_at


def _point_at_snr(
    model: StressNet, snr: float | None, *, n_traces: int, window_count: int,
    base_seed: int, snr_idx: int, detector_factory: Callable[[], StressDetector],
) -> DetectorSNRPoint:
    pipe = HostStressPipeline(model, detector_factory())

    # stressed traces: detection + latency
    detected = 0
    latencies: list[int] = []
    for t in range(n_traces):
        gen = torch.Generator().manual_seed(base_seed + snr_idx * 1000 + t)
        at = _run_trace(pipe, True, snr, window_count, gen)
        if at is not None:
            detected += 1
            latencies.append(at)

    # calm traces: false alarms (offset the seed space so draws don't overlap)
    false_alarms = 0
    for t in range(n_traces):
        gen = torch.Generator().manual_seed(base_seed + snr_idx * 1000 + 500 + t)
        at = _run_trace(pipe, False, snr, window_count, gen)
        if at is not None:
            false_alarms += 1

    return DetectorSNRPoint(
        snr_db=snr,
        detect_rate=round(detected / max(1, n_traces), 4),
        false_alarm_rate=round(false_alarms / max(1, n_traces), 4),
        median_latency_windows=(
            round(statistics.median(latencies), 2) if latencies else None
        ),
        n_traces=n_traces,
    )


def detector_robustness(
    model: StressNet,
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_traces: int = 32,
    window_count: int = 8,
    seed: int = 1,
    detector_factory: Callable[[], StressDetector] = StressDetector,
    detect_target: float = 0.8,
    fa_tolerance: float = 0.2,
    out_dir: str | Path | None = None,
) -> DetectorRobustnessResult:
    """Stream noisy traces through the detector; report per-SNR detection stats.

    Each trace is ``window_count`` consecutive windows of one class at a fixed
    SNR, run through a *fresh* detector (EMA/latch reset per trace). The clean
    point is first by convention so the scan runs least→most noise.
    """
    model = model.eval()
    points = [
        _point_at_snr(
            model, snr, n_traces=n_traces, window_count=window_count,
            base_seed=seed, snr_idx=i, detector_factory=detector_factory,
        )
        for i, snr in enumerate(snr_levels)
    ]

    floor: float | None = None
    for p in points:
        if p.snr_db is None:
            continue
        if p.detect_rate >= detect_target and p.false_alarm_rate <= fa_tolerance:
            floor = p.snr_db
        else:
            break  # scanning clean->noisy: stop at the first SNR that fails

    out = DetectorRobustnessResult(
        points=points, window_count=window_count, detect_target=detect_target,
        fa_tolerance=fa_tolerance, detection_floor_db=floor,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "detector_robustness.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "detector_robustness.md").write_text(to_markdown(out))

    return out


def _latency_label(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def to_markdown(out: DetectorRobustnessResult) -> str:
    if out.detection_floor_db is None:
        verdict = (
            f"**No noisy detection floor** — below clean audio the detector "
            f"misses the {out.detect_target:.0%} detection target (or false-alarms "
            f"past {out.fa_tolerance:.0%}). Treat noisy regimes as low-confidence."
        )
    else:
        verdict = (
            f"**Detection floor: {_snr_label(out.detection_floor_db)}** — the "
            f"detector still latches stress on ≥{out.detect_target:.0%} of stressed "
            f"traces while false-alarming on ≤{out.fa_tolerance:.0%} of calm ones "
            "down to this SNR."
        )
    header = (
        "# End-to-end detector robustness under noise\n\n"
        f"Multi-window traces ({out.window_count} windows each) are streamed "
        "through the EMA + hysteresis detector — a fresh detector per trace — at "
        "each SNR. `detect rate` is on stressed traces, `false alarm` on calm "
        "ones, `latency` is the median windows-to-latch on detected traces.\n\n"
        f"- {verdict}\n\n"
        "| SNR | detect rate | false alarm | latency (windows) |\n"
        "|---|---|---|---|\n"
    )
    rows = [
        f"| {_snr_label(p.snr_db)} | {p.detect_rate:.3f} | {p.false_alarm_rate:.3f} "
        f"| {_latency_label(p.median_latency_windows)} |"
        for p in out.points
    ]
    return header + "\n".join(rows) + "\n"


def build_detector_robustness(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_traces: int = 32,
    window_count: int = 8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> DetectorRobustnessResult:
    """Train the production model and stream-test its detector under noise."""
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed, snr_levels=snr_levels,
    )
    return detector_robustness(
        model, snr_levels=snr_levels, n_traces=n_traces,
        window_count=window_count, seed=seed + 1, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end detector robustness sweep")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--n-traces", type=int, default=32)
    ap.add_argument("--window-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_detector_robustness(
        epochs=args.epochs, n_per_class=args.n_per_class, seed=args.seed,
        n_traces=args.n_traces, window_count=args.window_count, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/detector_robustness.json and detector_robustness.md")


if __name__ == "__main__":
    main()
