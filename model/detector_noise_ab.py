"""A/B the detector knobs *in the noise regime* — can a faster attack hear
through the noise the default detector goes silent in?

Two measured findings set this up:

* ``noise_failure_mode.py`` — under noise the model fails by going **silent**
  (recall → 0, false positives stay near zero). It does not cry wolf.
* ``detector_robustness.py`` — the shipped detector's end-to-end **detection
  floor is -5 dB**; below it stressed traces stop latching.

Both point at one hypothesis: because the model isn't false-alarming in noise,
a **lower stress threshold / faster attack (higher EMA α)** should recover
detections at the noisy edge at little false-alarm cost. The clean-audio tuning
in ``tune_detector.py`` can't see this — it sweeps abstract score traces, not
the real model under acoustic noise. So this harness runs the *same* model and
the *same* noisy traces through ``detector_robustness`` once per candidate
config and reports, per config, the detection floor, the worst-case false-alarm
rate, and the mean detection rate — then recommends the config that pushes the
floor deepest into noise while keeping worst-case false alarms within tolerance.

On the brittle clean/10-dB-recipe model this hypothesis was *refuted* — lowering
the threshold tripped false alarms on the noise that hid stress, so the default
won. Re-run on the production **aggressive recipe** the picture inverts: the
model's scores under noise are now clean enough that every config holds the
-5 dB floor at *zero* false alarms, so the recommendation flips to the
lower-threshold config (a faster, more sensitive gate now costs nothing). The
robustness lever was training the model, not the knobs — and once the model is
robust, the knobs are free to be more sensitive.

    python -m model.detector_noise_ab

Records ``docs/benchmarks/detector_noise_ab.{json,md}``. Host-only; no device,
no AI Hub token.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .detector import StressDetector
from .detector_robustness import detector_robustness
from .model import StressNet
from .robustness import _snr_label

__all__ = [
    "DetectorConfig",
    "ConfigResult",
    "DetectorNoiseABResult",
    "detector_noise_ab",
    "build_detector_noise_ab",
]


@dataclass(frozen=True)
class DetectorConfig:
    """One candidate detector operating point."""

    label: str
    stress_threshold: float
    release_threshold: float
    ema_alpha: float

    def factory(self):
        """A zero-arg builder of a fresh detector at this operating point."""
        return lambda: StressDetector(
            stress_threshold=self.stress_threshold,
            release_threshold=self.release_threshold,
            ema_alpha=self.ema_alpha,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConfigResult:
    """How one config did across the noise sweep."""

    config: DetectorConfig
    detection_floor_db: float | None      # deeper (more negative) is better
    worst_false_alarm_rate: float         # max over SNRs — the safety budget
    mean_detect_rate: float               # mean over SNRs — overall sensitivity
    mean_latency_windows: float | None    # mean median-latency where it fired
    points: list[dict]                    # the per-SNR detector_robustness rows

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


@dataclass(frozen=True)
class DetectorNoiseABResult:
    results: list[ConfigResult]
    recommended: ConfigResult | None
    fa_tolerance: float

    def to_dict(self) -> dict:
        return {
            "fa_tolerance": self.fa_tolerance,
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "results": [r.to_dict() for r in self.results],
        }


# A "no noisy floor" config still needs to sort *worse* than any real floor, so
# map None to +inf when ranking by floor (we want the minimum / deepest floor).
_NO_FLOOR = float("inf")


def _evaluate(
    model: StressNet, cfg: DetectorConfig, *, snr_levels, n_traces: int,
    window_count: int, seed: int, detect_target: float, fa_tolerance: float,
) -> ConfigResult:
    sweep = detector_robustness(
        model, snr_levels=snr_levels, n_traces=n_traces, window_count=window_count,
        seed=seed, detector_factory=cfg.factory(), detect_target=detect_target,
        fa_tolerance=fa_tolerance,
    )
    pts = sweep.points
    worst_fa = max((p.false_alarm_rate for p in pts), default=0.0)
    mean_detect = round(statistics.mean(p.detect_rate for p in pts), 4) if pts else 0.0
    lat = [p.median_latency_windows for p in pts if p.median_latency_windows is not None]
    return ConfigResult(
        config=cfg,
        detection_floor_db=sweep.detection_floor_db,
        worst_false_alarm_rate=round(worst_fa, 4),
        mean_detect_rate=mean_detect,
        mean_latency_windows=round(statistics.mean(lat), 2) if lat else None,
        points=[p.to_dict() for p in pts],
    )


def detector_noise_ab(
    model: StressNet,
    configs: list[DetectorConfig],
    *,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_traces: int = 32,
    window_count: int = 8,
    seed: int = 1,
    detect_target: float = 0.8,
    fa_tolerance: float = 0.2,
    out_dir: str | Path | None = None,
) -> DetectorNoiseABResult:
    """Sweep detector configs through the noise robustness harness; recommend one.

    Every config sees the *same* model and the *same* seeded noisy traces, so
    differences are the knobs alone. The recommendation, among configs whose
    worst-case false-alarm rate is within ``fa_tolerance``, has the **deepest
    detection floor** (recovers furthest into noise); ties break toward lower
    latency, then higher mean detection rate.
    """
    model = model.eval()
    results = [
        _evaluate(
            model, cfg, snr_levels=snr_levels, n_traces=n_traces,
            window_count=window_count, seed=seed, detect_target=detect_target,
            fa_tolerance=fa_tolerance,
        )
        for cfg in configs
    ]

    eligible = [r for r in results if r.worst_false_alarm_rate <= fa_tolerance]
    recommended = min(
        eligible,
        key=lambda r: (
            r.detection_floor_db if r.detection_floor_db is not None else _NO_FLOOR,
            r.mean_latency_windows if r.mean_latency_windows is not None else _NO_FLOOR,
            -r.mean_detect_rate,
        ),
        default=None,
    )

    out = DetectorNoiseABResult(
        results=results, recommended=recommended, fa_tolerance=fa_tolerance,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "detector_noise_ab.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "detector_noise_ab.md").write_text(to_markdown(out))

    return out


def _floor_label(v: float | None) -> str:
    return "none (clean only)" if v is None else _snr_label(v)


def to_markdown(out: DetectorNoiseABResult) -> str:
    rec = out.recommended
    if rec is None:
        verdict = (
            f"**No config stays within the {out.fa_tolerance:.0%} false-alarm "
            "budget** — every candidate false-alarms too much somewhere in the "
            "sweep. Keep the default and treat noisy regimes as low-confidence."
        )
    else:
        verdict = (
            f"**Recommended: `{rec.config.label}`** "
            f"(stress={rec.config.stress_threshold}, "
            f"release={rec.config.release_threshold}, "
            f"alpha={rec.config.ema_alpha}) — detection floor "
            f"{_floor_label(rec.detection_floor_db)}, worst false alarm "
            f"{rec.worst_false_alarm_rate:.3f} (≤ {out.fa_tolerance:.0%} budget)."
        )
    header = (
        "# Detector A/B in the noise regime\n\n"
        "Each detector config is run over the *same* model and the *same* noisy "
        "traces (via the end-to-end `detector_robustness` sweep). The winner "
        "pushes the **detection floor** deepest into noise while keeping the "
        "worst-case false-alarm rate within budget — testing whether a faster "
        "attack hears through the noise the default detector goes silent in.\n\n"
        f"- {verdict}\n\n"
        "| config | stress | release | alpha | detection floor | worst FA | "
        "mean detect | mean latency |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in out.results:
        star = " ⭐" if r is rec else ""
        c = r.config
        lat = "—" if r.mean_latency_windows is None else f"{r.mean_latency_windows:g}"
        rows.append(
            f"| `{c.label}`{star} | {c.stress_threshold} | {c.release_threshold} "
            f"| {c.ema_alpha} | {_floor_label(r.detection_floor_db)} "
            f"| {r.worst_false_alarm_rate:.3f} | {r.mean_detect_rate:.3f} | {lat} |"
        )
    return header + "\n".join(rows) + "\n"


# The default operating point plus faster-attack candidates the findings suggest.
_DEFAULT_GRID = [
    DetectorConfig("default", 0.6, 0.45, 0.4),
    DetectorConfig("lower-threshold", 0.5, 0.4, 0.4),
    DetectorConfig("fast-attack", 0.5, 0.4, 1.0),
    DetectorConfig("aggressive", 0.45, 0.35, 1.0),
]


def build_detector_noise_ab(
    *,
    epochs: int = 12,
    n_per_class: int = 96,
    seed: int = 0,
    configs: list[DetectorConfig] | None = None,
    snr_levels: list[float | None] = (None, 20.0, 10.0, 0.0, -5.0, -10.0),
    n_traces: int = 32,
    window_count: int = 8,
    out_dir: str | Path | None = "docs/benchmarks",
) -> DetectorNoiseABResult:
    """Train the production model and A/B the detector knobs under noise."""
    from .production import train_production

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed, snr_levels=snr_levels,
    )
    return detector_noise_ab(
        model, configs if configs is not None else _DEFAULT_GRID,
        snr_levels=snr_levels, n_traces=n_traces, window_count=window_count,
        seed=seed + 1, out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B the detector knobs under noise")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--n-traces", type=int, default=32)
    ap.add_argument("--window-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = build_detector_noise_ab(
        epochs=args.epochs, n_per_class=args.n_per_class, seed=args.seed,
        n_traces=args.n_traces, window_count=args.window_count, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/detector_noise_ab.json and detector_noise_ab.md")


if __name__ == "__main__":
    main()
