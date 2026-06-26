"""Tune the on-device stress-detector UX knobs on the host.

The detector's behaviour is governed by three knobs (model/audio_config.py):
``STRESS_THRESHOLD`` / ``RELEASE_THRESHOLD`` (the hysteresis band) and
``EMA_ALPHA`` (smoothing). Picking them well is a trade-off: tighter thresholds
and less smoothing react faster but **flicker** at the boundary; looser/smoother
settings are steady but lag. This harness makes that trade-off measurable.

Given a labelled per-window score trace, it runs the host StressDetector (the
golden spec) under each candidate setting and scores it on two axes:

  - **accuracy** — fraction of windows whose latch matches ground truth, and
  - **flicker** — number of latch transitions (fewer = steadier meter).

A combined ``score = accuracy - flicker_weight * flicker/n`` ranks the grid and
picks a recommended operating point. Artifacts go to ``docs/benchmarks/``.

    python -m model.tune_detector            # sweep around the current defaults

Everything is host-side and deterministic; no device, no AI Hub, no live token.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from .audio_config import EMA_ALPHA, RELEASE_THRESHOLD, STRESS_THRESHOLD
from .detector import StressDetector


def synthetic_score_trace(
    *,
    n_segments: int = 6,
    seg_len: int = 12,
    seed: int = 0,
    noise: float = 0.15,
    calm_base: float = 0.25,
    stressed_base: float = 0.75,
) -> tuple[list[float], list[int], list[int]]:
    """A labelled score trace alternating calm/stressed segments.

    Segment 0 is calm (label 0); segments alternate. Each window's raw score is
    its segment base plus uniform ``±noise`` jitter, clamped to [0, 1]. All
    windows are voiced. Returns ``(scores, voiced, labels)``.
    """
    rng = random.Random(seed)
    scores: list[float] = []
    labels: list[int] = []
    for seg in range(n_segments):
        label = seg % 2  # 0 calm, 1 stressed
        base = stressed_base if label else calm_base
        for _ in range(seg_len):
            s = base + rng.uniform(-noise, noise)
            scores.append(min(1.0, max(0.0, s)))
            labels.append(label)
    voiced = [1] * len(scores)
    return scores, voiced, labels


@dataclass(frozen=True)
class ConfigMetrics:
    """How one (threshold, threshold, alpha) setting did on a trace."""

    stress_threshold: float
    release_threshold: float
    ema_alpha: float
    accuracy: float
    flicker: int
    n_windows: int
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_config(
    scores,
    voiced,
    labels,
    *,
    stress_threshold: float,
    release_threshold: float,
    ema_alpha: float,
    flicker_weight: float = 0.5,
) -> ConfigMetrics:
    """Run the detector under one setting and score accuracy + flicker."""
    det = StressDetector(
        stress_threshold=stress_threshold,
        release_threshold=release_threshold,
        ema_alpha=ema_alpha,
    )
    states = det.run(scores, [bool(v) for v in voiced])

    n = len(states)
    correct = sum(int(st.stressed) == int(lbl) for st, lbl in zip(states, labels))
    accuracy = correct / n if n else 0.0

    flicker = sum(
        1 for i in range(1, n) if states[i].stressed != states[i - 1].stressed
    )
    score = accuracy - flicker_weight * (flicker / n if n else 0.0)

    return ConfigMetrics(
        stress_threshold=stress_threshold,
        release_threshold=release_threshold,
        ema_alpha=ema_alpha,
        accuracy=round(accuracy, 4),
        flicker=flicker,
        n_windows=n,
        score=round(score, 4),
    )


@dataclass(frozen=True)
class SweepResult:
    results: list[ConfigMetrics]
    best: ConfigMetrics

    def to_dict(self) -> dict:
        return {
            "best": self.best.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }


def sweep(
    scores,
    voiced,
    labels,
    *,
    stress_thresholds,
    release_thresholds,
    ema_alphas,
    flicker_weight: float = 0.5,
    out_dir: str | Path | None = None,
) -> SweepResult:
    """Grid-search the knobs; rank by combined score; pick the best.

    Only valid bands (``release_threshold < stress_threshold``) are evaluated.
    The recommended config maximises ``score``, ties broken toward higher
    accuracy then fewer flickers.
    """
    results: list[ConfigMetrics] = []
    for st, rl, al in product(stress_thresholds, release_thresholds, ema_alphas):
        if not rl < st:
            continue
        results.append(
            evaluate_config(
                scores, voiced, labels,
                stress_threshold=st, release_threshold=rl, ema_alpha=al,
                flicker_weight=flicker_weight,
            )
        )
    if not results:
        raise ValueError("no valid (release < stress) configs in the grid")

    best = max(results, key=lambda m: (m.score, m.accuracy, -m.flicker))
    out = SweepResult(results=results, best=best)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "detector_tuning.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "detector_tuning.md").write_text(to_markdown(out))

    return out


def to_markdown(out: SweepResult) -> str:
    """Render the swept grid as a table, recommended row starred."""
    b = out.best
    header = (
        "# StressDetector tuning sweep\n\n"
        "Each setting is run over a labelled synthetic score trace through the "
        "host `StressDetector` (the golden spec). **accuracy** = latch matches "
        "truth; **flicker** = latch transitions (lower is steadier). "
        "`score = accuracy - 0.5·flicker/n`.\n\n"
        f"- current defaults: stress={STRESS_THRESHOLD}, "
        f"release={RELEASE_THRESHOLD}, alpha={EMA_ALPHA}\n"
        f"- **recommended: stress={b.stress_threshold}, "
        f"release={b.release_threshold}, alpha={b.ema_alpha}** "
        f"(acc {b.accuracy:.3f}, flicker {b.flicker})\n\n"
        "| stress | release | alpha | accuracy | flicker | score |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in sorted(out.results, key=lambda m: -m.score):
        star = " ⭐" if r is b else ""
        rows.append(
            f"| {r.stress_threshold}{star} | {r.release_threshold} | {r.ema_alpha} | "
            f"{r.accuracy:.3f} | {r.flicker} | {r.score:.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune the host StressDetector knobs")
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--n-segments", type=int, default=8)
    ap.add_argument("--seg-len", type=int, default=12)
    args = ap.parse_args()

    scores, voiced, labels = synthetic_score_trace(
        n_segments=args.n_segments, seg_len=args.seg_len, seed=args.seed, noise=args.noise
    )
    out = sweep(
        scores, voiced, labels,
        stress_thresholds=[0.55, 0.6, 0.65, 0.7],
        release_thresholds=[0.35, 0.4, 0.45, 0.5],
        ema_alphas=[0.2, 0.3, 0.4, 0.6, 1.0],
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/detector_tuning.json and detector_tuning.md")


if __name__ == "__main__":
    main()
