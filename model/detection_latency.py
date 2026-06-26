"""Detector onset/offset latency A/B (time-to-alarm).

Every existing detector artifact (``detector_tuning``, ``detector_robustness``,
``detector_noise_ab``) scores per-window *accuracy* or flicker. None measures the
detector's *temporal response*. The shipped ``StressDetector`` runs an EMA
(alpha 0.4) under dual-threshold hysteresis (enter 0.6 / release 0.45): that
smoothing is what stops the latch flickering, but it also means the alarm fires
several hop-windows *after* stress actually begins and lingers after it ends.
With ``HOP_SECONDS = 1.0`` each window is a second of wall-clock, so this delay
is exactly the product-visible "how long until it notices / how long until it
lets go" number — distinct from ``latency_rtf`` (which times the *compute* of one
window, not the *detection* delay across windows).

This A/B drives the real detector through synthetic calm->stressed->calm
episodes and reports, per episode, the **onset latency** (windows/seconds from
the true stress onset until the latch fires), whether it fired at all
(**detection rate**), and the **release latency** after stress ends. A
responsive detector fires within a small budget on every episode.

Host-only: CPU training + the pure-Python detector, no device, no AI Hub token.
The reduction is pure (unit-tested with hand-set latencies);
``build_detection_latency`` trains one tiny net and runs real episodes.

Run:
    PYTHONPATH=. .venv/bin/python -m model.detection_latency
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

__all__ = [
    "DetectionEpisode",
    "DetectionLatencyResult",
    "detection_latency",
    "build_detection_latency",
]

# Default seconds-to-alarm we consider "responsive". The window is 3 s and the
# EMA needs a couple of hops to cross 0.6, so a few seconds is the design target.
_DEFAULT_ONSET_BUDGET_S = 3.0


@dataclass(frozen=True)
class DetectionEpisode:
    """One calm->stressed->calm episode's detector response.

    ``onset_latency_windows`` is the number of hop-windows from the true stress
    onset until the latch first reads stressed (``None`` = never fired during the
    episode — a miss). ``release_latency_windows`` is windows from the true
    offset until the latch clears (``None`` = stayed latched — never released).
    """

    onset_latency_windows: int | None
    release_latency_windows: int | None
    hop_seconds: float

    @property
    def detected(self) -> bool:
        return self.onset_latency_windows is not None

    @property
    def released(self) -> bool:
        return self.release_latency_windows is not None

    @property
    def onset_latency_s(self) -> float | None:
        if self.onset_latency_windows is None:
            return None
        return self.onset_latency_windows * self.hop_seconds

    @property
    def release_latency_s(self) -> float | None:
        if self.release_latency_windows is None:
            return None
        return self.release_latency_windows * self.hop_seconds

    def to_dict(self) -> dict:
        return {
            "onset_latency_windows": self.onset_latency_windows,
            "release_latency_windows": self.release_latency_windows,
            "onset_latency_s": self.onset_latency_s,
            "release_latency_s": self.release_latency_s,
            "detected": self.detected,
            "released": self.released,
        }


@dataclass(frozen=True)
class DetectionLatencyResult:
    """Temporal-response reduction over a set of episodes."""

    episodes: list[DetectionEpisode]
    hop_seconds: float
    onset_budget_s: float

    @property
    def _detected(self) -> list[DetectionEpisode]:
        return [e for e in self.episodes if e.detected]

    @property
    def detection_rate(self) -> float:
        return len(self._detected) / len(self.episodes)

    @property
    def median_onset_windows(self) -> float | None:
        det = self._detected
        return median(e.onset_latency_windows for e in det) if det else None

    @property
    def median_onset_s(self) -> float | None:
        m = self.median_onset_windows
        return None if m is None else m * self.hop_seconds

    @property
    def max_onset_s(self) -> float | None:
        det = self._detected
        return max(e.onset_latency_s for e in det) if det else None

    @property
    def release_rate(self) -> float | None:
        """Fraction of *detected* episodes that eventually released the latch."""
        det = self._detected
        if not det:
            return None
        return sum(1 for e in det if e.released) / len(det)

    @property
    def median_release_s(self) -> float | None:
        rel = [e.release_latency_s for e in self._detected if e.released]
        return median(rel) if rel else None

    @property
    def responsive(self) -> bool:
        """Fires on every episode, with median onset within the budget."""
        if self.detection_rate < 1.0:
            return False
        m = self.median_onset_s
        return m is not None and m <= self.onset_budget_s

    @property
    def verdict(self) -> str:
        dr = self.detection_rate
        if self.responsive:
            rel = self.median_release_s
            rel_txt = f"{rel:g}s" if rel is not None else "—"
            return (
                f"**Responsive**: the detector fires on every episode with a "
                f"median onset latency of {self.median_onset_s:g}s "
                f"(worst {self.max_onset_s:g}s, budget {self.onset_budget_s:g}s) "
                f"and releases a median {rel_txt} after stress ends. The EMA + "
                "hysteresis smoothing buys flicker-freedom without making the "
                "alarm feel sluggish."
            )
        if dr < 1.0:
            return (
                f"**Misses onsets**: the latch fired on only {dr:.0%} of "
                "episodes — on the rest the smoothed score never crossed the "
                "0.6 enter threshold. Lower the enter threshold or raise EMA "
                "alpha so genuine stress onsets aren't smoothed away."
            )
        return (
            f"**Sluggish**: the detector fires on every episode but the median "
            f"onset latency is {self.median_onset_s:g}s (> {self.onset_budget_s:g}s "
            "budget). The EMA is smoothing too hard — raise alpha or lower the "
            "enter threshold to shorten time-to-alarm."
        )

    def to_dict(self) -> dict:
        return {
            "hop_seconds": self.hop_seconds,
            "onset_budget_s": self.onset_budget_s,
            "n_episodes": len(self.episodes),
            "detection_rate": self.detection_rate,
            "median_onset_windows": self.median_onset_windows,
            "median_onset_s": self.median_onset_s,
            "max_onset_s": self.max_onset_s,
            "release_rate": self.release_rate,
            "median_release_s": self.median_release_s,
            "responsive": self.responsive,
            "verdict": self.verdict,
            "episodes": [e.to_dict() for e in self.episodes],
        }


def detection_latency(
    records,
    *,
    hop_seconds: float,
    onset_budget_s: float = _DEFAULT_ONSET_BUDGET_S,
    out_dir: str | Path | None = None,
) -> DetectionLatencyResult:
    """Reduce per-episode ``(onset_latency_windows, release_latency_windows)``.

    Either latency may be ``None`` (missed onset / never released). Pure — no
    training, no device. ``hop_seconds`` must be positive; present latencies must
    be non-negative. Writes ``detection_latency.{json,md}`` to ``out_dir`` when
    given.
    """
    if hop_seconds <= 0.0:
        raise ValueError("hop_seconds must be positive")

    episodes = []
    for onset, release in records:
        if onset is not None and int(onset) < 0:
            raise ValueError("onset latency must be non-negative")
        if release is not None and int(release) < 0:
            raise ValueError("release latency must be non-negative")
        episodes.append(
            DetectionEpisode(
                onset_latency_windows=None if onset is None else int(onset),
                release_latency_windows=None if release is None else int(release),
                hop_seconds=hop_seconds,
            )
        )
    if not episodes:
        raise ValueError("records must be non-empty")

    out = DetectionLatencyResult(
        episodes=episodes, hop_seconds=hop_seconds, onset_budget_s=onset_budget_s
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "detection_latency.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "detection_latency.md").write_text(to_markdown(out))

    return out


def _fmt(x: float | None, unit: str = "") -> str:
    return "—" if x is None else f"{x:g}{unit}"


def to_markdown(out: DetectionLatencyResult) -> str:
    header = (
        "# Detector onset/offset latency (A/B)\n\n"
        "The EMA + dual-threshold hysteresis that keeps the latch from flickering "
        "also delays it. This drives the real detector through "
        "calm→stressed→calm episodes and measures **time-to-alarm**: with "
        f"`HOP_SECONDS`={out.hop_seconds:g}s each window is a second of "
        "wall-clock.\n\n"
        f"- detection rate: **{out.detection_rate:.0%}** "
        f"({len(out.episodes)} episodes)\n"
        f"- median onset latency: **{_fmt(out.median_onset_s, 's')}** "
        f"(worst {_fmt(out.max_onset_s, 's')}, budget {out.onset_budget_s:g}s)\n"
        f"- release: **{_fmt(out.release_rate and out.release_rate * 100, '%')}** "
        f"released, median **{_fmt(out.median_release_s, 's')}** after offset\n"
        f"- responsive: **{out.responsive}**\n"
        f"- {out.verdict}\n\n"
        "| episode | onset (win) | onset (s) | released | release (s) |\n"
        "|---|---|---|---|---|\n"
    )
    rows = [
        f"| {i} | {_fmt(e.onset_latency_windows)} | {_fmt(e.onset_latency_s, 's')} "
        f"| {e.released} | {_fmt(e.release_latency_s, 's')} |"
        for i, e in enumerate(out.episodes)
    ]
    return header + "\n".join(rows) + "\n"


def build_detection_latency(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    n_episodes: int = 4,
    calm_windows: int = 4,
    stress_windows: int = 12,
    tail_windows: int = 12,
    onset_budget_s: float = _DEFAULT_ONSET_BUDGET_S,
    out_dir: str | Path | None = "docs/benchmarks",
) -> DetectionLatencyResult:
    """Train one net, drive the real detector through stress episodes.

    Each episode is ``calm_windows`` calm windows, then ``stress_windows``
    stressed windows (the true onset), then ``tail_windows`` calm windows (the
    true offset). Per-window model scores feed the shipped ``StressDetector``;
    onset latency is the first stressed-latch index minus the true onset index,
    release latency the first clear index after the offset. Imports inside the
    function to keep torch/training off the pure reduction path.
    """
    import torch

    from .audio_config import HOP_SECONDS
    from .detector import StressDetector
    from .production import train_production
    from .robustness import noisy_synthetic_dataset

    model, _ = train_production(
        epochs=epochs, n_per_class=n_per_class, seed=seed,
        snr_levels=snr_levels, threshold=0.8,
    )
    model = model.eval()

    def _scores(stressed: bool, k: int, snr, ep_seed: int):
        """k per-window scores for the chosen class at this SNR."""
        # noisy_synthetic_dataset returns a balanced batch; take the class rows.
        x, y = noisy_synthetic_dataset(max(k, 1), ep_seed, snr_db=snr)
        sel = x[(y.flatten() == (1 if stressed else 0))]
        if sel.shape[0] < k:  # pad by cycling if the class had fewer rows
            reps = (k + sel.shape[0] - 1) // sel.shape[0]
            sel = sel.repeat(reps, 1, 1, 1)
        with torch.no_grad():
            return model(sel[:k]).flatten().tolist()

    records = []
    for snr in snr_levels:
        for e in range(n_episodes):
            ep_seed = seed + 1000 + 7 * e
            calm = _scores(False, calm_windows, snr, ep_seed)
            stress = _scores(True, stress_windows, snr, ep_seed + 1)
            tail = _scores(False, tail_windows, snr, ep_seed + 2)
            seq = calm + stress + tail
            states = StressDetector().run(seq, [True] * len(seq))
            onset_idx = calm_windows
            offset_idx = calm_windows + stress_windows

            onset_lat = None
            for i in range(onset_idx, len(states)):
                if states[i].stressed:
                    onset_lat = i - onset_idx
                    break
            release_lat = None
            if onset_lat is not None:
                for j in range(offset_idx, len(states)):
                    if not states[j].stressed:
                        release_lat = max(0, j - offset_idx)
                        break
            records.append((onset_lat, release_lat))

    return detection_latency(
        records, hop_seconds=HOP_SECONDS, onset_budget_s=onset_budget_s,
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detector onset/offset latency A/B (time-to-alarm)"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    ap.add_argument("--n-episodes", type=int, default=4)
    args = ap.parse_args()
    out = build_detection_latency(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        n_episodes=args.n_episodes, out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/detection_latency.json and .md")


if __name__ == "__main__":
    main()
