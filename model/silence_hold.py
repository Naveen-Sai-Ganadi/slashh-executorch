"""Detector silence-hold / gap robustness A/B.

Every existing detector artifact (``detector_tuning``, ``detector_robustness``,
``detector_noise_ab``, ``detection_latency``) drives the detector with a
*continuous* voiced stream — ``voiced=True`` on every window. None exercises the
VAD-gated path. On an unvoiced window the shipped ``StressDetector`` does not run
the model: it **freezes the EMA and holds the latch** (see
``model/detector.py``, the ``if not voiced`` branch). That gating has a
product-visible consequence: if the user falls silent right after a stress
episode ends, the alarm stays latched for the *whole* silence, because the EMA —
which is what eventually drops the latch — only decays on voiced windows. With
``HOP_SECONDS = 1.0`` every gap window is a second the alarm lingers.

This A/B drives the real detector through calm→stress(latch on)→**silence
gap**→calm episodes and measures, per episode, how long after the true stress
offset the latch clears, counted two ways:

- **voiced release** — voiced windows after offset until release. This is the
  EMA decay budget and is *independent* of how long the silence lasts.
- **wall-clock release** — total windows (voiced + the unvoiced gap) until
  release. This is what the user actually experiences.

Their difference is the **silence inflation**: seconds the alarm persists purely
because the user went quiet. A latch that holds through silence inflates by
exactly the gap length; one that dropped mid-silence would inflate by less.
Whether "alarm persists through silence" is a feature (don't forget ongoing
stress just because of a pause) or a nuisance (stale alarm long after the
moment passed) is a product call — this artifact quantifies it either way.

Host-only: CPU training + the pure-Python detector, no device, no AI Hub token.
The reduction is pure (unit-tested with hand-set records); ``build_silence_hold``
trains one tiny net and runs real gap episodes.

Run:
    PYTHONPATH=. .venv/bin/python -m model.silence_hold
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

__all__ = [
    "SilenceHoldEpisode",
    "SilenceHoldResult",
    "silence_hold",
    "build_silence_hold",
]

# Default seconds we allow the alarm to take to clear after stress ends. The
# voiced EMA decay is a couple of windows; a long silence on top of that is the
# behaviour this budget flags.
_DEFAULT_CLEAR_BUDGET_S = 8.0


@dataclass(frozen=True)
class SilenceHoldEpisode:
    """One calm→stress→silence→calm episode's latch-clear behaviour.

    ``voiced_release_windows`` counts only voiced windows after the true stress
    offset until the latch clears (the EMA decay budget). ``wallclock_release_
    windows`` counts every window including the unvoiced gap. Both are ``None``
    together when the latch never released during the episode.
    """

    gap_windows: int
    voiced_release_windows: int | None
    wallclock_release_windows: int | None
    hop_seconds: float

    @property
    def released(self) -> bool:
        return self.wallclock_release_windows is not None

    @property
    def voiced_release_s(self) -> float | None:
        if self.voiced_release_windows is None:
            return None
        return self.voiced_release_windows * self.hop_seconds

    @property
    def wallclock_release_s(self) -> float | None:
        if self.wallclock_release_windows is None:
            return None
        return self.wallclock_release_windows * self.hop_seconds

    @property
    def inflation_windows(self) -> int | None:
        """Wall-clock minus voiced release windows — the silence-added delay."""
        if not self.released:
            return None
        return self.wallclock_release_windows - self.voiced_release_windows

    @property
    def inflation_s(self) -> float | None:
        iw = self.inflation_windows
        return None if iw is None else iw * self.hop_seconds

    def to_dict(self) -> dict:
        return {
            "gap_windows": self.gap_windows,
            "voiced_release_windows": self.voiced_release_windows,
            "wallclock_release_windows": self.wallclock_release_windows,
            "voiced_release_s": self.voiced_release_s,
            "wallclock_release_s": self.wallclock_release_s,
            "inflation_windows": self.inflation_windows,
            "inflation_s": self.inflation_s,
            "released": self.released,
        }


@dataclass(frozen=True)
class SilenceHoldResult:
    """Silence-gap reduction over a set of episodes."""

    episodes: list[SilenceHoldEpisode]
    hop_seconds: float
    clear_budget_s: float

    @property
    def _released(self) -> list[SilenceHoldEpisode]:
        return [e for e in self.episodes if e.released]

    @property
    def release_rate(self) -> float:
        return len(self._released) / len(self.episodes)

    @property
    def median_voiced_release_s(self) -> float | None:
        rel = self._released
        return median(e.voiced_release_s for e in rel) if rel else None

    @property
    def median_wallclock_release_s(self) -> float | None:
        rel = self._released
        return median(e.wallclock_release_s for e in rel) if rel else None

    @property
    def max_wallclock_release_s(self) -> float | None:
        rel = self._released
        return max(e.wallclock_release_s for e in rel) if rel else None

    @property
    def median_inflation_s(self) -> float | None:
        rel = self._released
        return median(e.inflation_s for e in rel) if rel else None

    @property
    def max_inflation_s(self) -> float | None:
        rel = self._released
        return max(e.inflation_s for e in rel) if rel else None

    @property
    def holds_through_silence(self) -> bool:
        """Does the latch persist across the *entire* silence on every gap?

        True when each released episode that actually had a gap inflated by
        exactly its gap length — i.e. the latch stayed on for the whole quiet
        stretch and only decayed on the voiced windows that followed. False if
        any gapped episode dropped the latch mid-silence (inflation < gap).
        """
        gapped = [e for e in self._released if e.gap_windows > 0]
        if not gapped:
            return True  # nothing to violate
        return all(e.inflation_windows == e.gap_windows for e in gapped)

    @property
    def clears_within_budget(self) -> bool:
        """Every episode releases, and within the wall-clock clear budget."""
        rel = self._released
        if not rel or len(rel) != len(self.episodes):
            return False
        return all(e.wallclock_release_s <= self.clear_budget_s for e in rel)

    @property
    def verdict(self) -> str:
        if not self._released:
            return (
                "**Inconclusive**: no episode produced a measurable latch "
                "release — the latch either never engaged or never cleared, so "
                "silence-hold behaviour can't be characterized. Lengthen the "
                "stress/tail phases."
            )
        worst = self.max_inflation_s
        worst_txt = f"{worst:g}s" if worst is not None else "—"
        if self.holds_through_silence and self.clears_within_budget:
            return (
                "**Holds through silence, clears in budget**: the latch persists "
                "across the entire quiet stretch (inflation tracks the gap "
                f"exactly, worst {worst_txt}) yet every episode still clears "
                f"within the {self.clear_budget_s:g}s budget. The VAD gate keeps "
                "ongoing stress latched through pauses without leaving a stale "
                "alarm — the intended behaviour."
            )
        if self.holds_through_silence:
            return (
                "**Holds through silence, exceeds budget**: the latch correctly "
                "persists through pauses, but a long silence keeps the alarm "
                f"latched for up to {self.max_wallclock_release_s:g}s after stress "
                f"ends (> {self.clear_budget_s:g}s budget) — the alarm lingers for "
                "the whole quiet stretch. If a pause should clear the meter, add a "
                "decay-on-silence or a max-hold timeout to the gated path."
            )
        return (
            "**Drops during silence**: on at least one episode the latch cleared "
            "*inside* the silence gap (inflation < gap), so ongoing stress that "
            "resumes after a pause would be missed. The EMA isn't actually frozen "
            "across the gap as intended — check the unvoiced branch of the "
            "detector."
        )

    def to_dict(self) -> dict:
        return {
            "hop_seconds": self.hop_seconds,
            "clear_budget_s": self.clear_budget_s,
            "n_episodes": len(self.episodes),
            "release_rate": self.release_rate,
            "median_voiced_release_s": self.median_voiced_release_s,
            "median_wallclock_release_s": self.median_wallclock_release_s,
            "max_wallclock_release_s": self.max_wallclock_release_s,
            "median_inflation_s": self.median_inflation_s,
            "max_inflation_s": self.max_inflation_s,
            "holds_through_silence": self.holds_through_silence,
            "clears_within_budget": self.clears_within_budget,
            "verdict": self.verdict,
            "episodes": [e.to_dict() for e in self.episodes],
        }


def silence_hold(
    records,
    *,
    hop_seconds: float,
    clear_budget_s: float = _DEFAULT_CLEAR_BUDGET_S,
    out_dir: str | Path | None = None,
) -> SilenceHoldResult:
    """Reduce per-episode ``(gap_windows, voiced_release, wallclock_release)``.

    ``voiced_release`` and ``wallclock_release`` are either both ``None`` (the
    latch never cleared) or both non-negative ints with
    ``wallclock_release >= voiced_release`` (the gap can only add windows). Pure —
    no training, no device. ``hop_seconds`` must be positive. Writes
    ``silence_hold.{json,md}`` to ``out_dir`` when given.
    """
    if hop_seconds <= 0.0:
        raise ValueError("hop_seconds must be positive")

    episodes = []
    for gap, voiced_rel, wall_rel in records:
        if int(gap) < 0:
            raise ValueError("gap_windows must be non-negative")
        if (voiced_rel is None) != (wall_rel is None):
            raise ValueError(
                "voiced_release and wallclock_release must both be None or both set"
            )
        if voiced_rel is not None:
            if int(voiced_rel) < 0 or int(wall_rel) < 0:
                raise ValueError("release windows must be non-negative")
            if int(wall_rel) < int(voiced_rel):
                raise ValueError(
                    "wallclock_release cannot be fewer windows than voiced_release"
                )
        episodes.append(
            SilenceHoldEpisode(
                gap_windows=int(gap),
                voiced_release_windows=None if voiced_rel is None else int(voiced_rel),
                wallclock_release_windows=None if wall_rel is None else int(wall_rel),
                hop_seconds=hop_seconds,
            )
        )
    if not episodes:
        raise ValueError("records must be non-empty")

    out = SilenceHoldResult(
        episodes=episodes, hop_seconds=hop_seconds, clear_budget_s=clear_budget_s
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "silence_hold.json").write_text(
            json.dumps(out.to_dict(), indent=2) + "\n"
        )
        (out_dir / "silence_hold.md").write_text(to_markdown(out))

    return out


def _fmt(x: float | None, unit: str = "") -> str:
    return "—" if x is None else f"{x:g}{unit}"


def to_markdown(out: SilenceHoldResult) -> str:
    header = (
        "# Detector silence-hold / gap robustness (A/B)\n\n"
        "On an unvoiced window the detector freezes its EMA and holds the latch, "
        "so a silence right after a stress episode keeps the alarm latched for the "
        "whole quiet stretch. This drives the real detector through "
        "calm→stress→**silence gap**→calm episodes and measures how long the latch "
        f"takes to clear after stress ends — in voiced windows (EMA decay budget) "
        f"vs wall-clock (with the gap). `HOP_SECONDS`={out.hop_seconds:g}s.\n\n"
        f"- release rate: **{out.release_rate:.0%}** ({len(out.episodes)} episodes)\n"
        f"- median voiced-only clear: **{_fmt(out.median_voiced_release_s, 's')}** "
        f"(gap-independent)\n"
        f"- median wall-clock clear: **{_fmt(out.median_wallclock_release_s, 's')}** "
        f"(worst {_fmt(out.max_wallclock_release_s, 's')})\n"
        f"- silence inflation: median **{_fmt(out.median_inflation_s, 's')}**, "
        f"worst **{_fmt(out.max_inflation_s, 's')}**\n"
        f"- holds through silence: **{out.holds_through_silence}**; clears within "
        f"{out.clear_budget_s:g}s budget: **{out.clears_within_budget}**\n"
        f"- {out.verdict}\n\n"
        "| episode | gap (s) | voiced clear (s) | wall-clock clear (s) "
        "| inflation (s) | released |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {i} | {_fmt(e.gap_windows * e.hop_seconds, 's')} "
        f"| {_fmt(e.voiced_release_s, 's')} | {_fmt(e.wallclock_release_s, 's')} "
        f"| {_fmt(e.inflation_s, 's')} | {e.released} |"
        for i, e in enumerate(out.episodes)
    ]
    return header + "\n".join(rows) + "\n"


def build_silence_hold(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    snr_levels=(None, 20.0, 10.0, 0.0, -5.0),
    gap_windows_list=(0, 3, 10),
    calm_pre: int = 3,
    stress_windows: int = 12,
    tail_windows: int = 20,
    clear_budget_s: float = _DEFAULT_CLEAR_BUDGET_S,
    out_dir: str | Path | None = "docs/benchmarks",
) -> SilenceHoldResult:
    """Train one net, drive the real detector through silence-gap episodes.

    Each episode is ``calm_pre`` calm voiced windows, then ``stress_windows``
    stressed voiced windows (latch turns on), then ``gap_windows`` **unvoiced**
    windows (the silence — EMA frozen, latch held), then ``tail_windows`` calm
    voiced windows (the EMA finally decays and the latch clears). The true stress
    offset is the start of the gap. Voiced release counts only the voiced tail
    windows until clear; wall-clock release counts the gap too. Imports inside the
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
        x, y = noisy_synthetic_dataset(max(k, 1), ep_seed, snr_db=snr)
        sel = x[(y.flatten() == (1 if stressed else 0))]
        if sel.shape[0] < k:  # pad by cycling if the class had fewer rows
            reps = (k + sel.shape[0] - 1) // sel.shape[0]
            sel = sel.repeat(reps, 1, 1, 1)
        with torch.no_grad():
            return model(sel[:k]).flatten().tolist()

    records = []
    e = 0
    for snr in snr_levels:
        for gap in gap_windows_list:
            ep_seed = seed + 1000 + 7 * e
            e += 1
            pre = _scores(False, calm_pre, snr, ep_seed)
            stress = _scores(True, stress_windows, snr, ep_seed + 1)
            tail = _scores(False, tail_windows, snr, ep_seed + 2)

            # Sequence: voiced pre + voiced stress + UNVOICED gap + voiced tail.
            seq = pre + stress + [0.0] * gap + tail
            voiced = (
                [True] * len(pre) + [True] * len(stress)
                + [False] * gap + [True] * len(tail)
            )
            states = StressDetector().run(seq, voiced)
            offset = len(pre) + len(stress)  # gap starts here = true stress offset

            engaged_at_offset = offset > 0 and states[offset - 1].stressed
            voiced_rel = None
            wall_rel = None
            if engaged_at_offset:
                vcount = 0
                for i in range(offset, len(states)):
                    if not states[i].stressed:
                        wall_rel = i - offset
                        voiced_rel = vcount
                        break
                    if voiced[i]:
                        vcount += 1
            records.append((gap, voiced_rel, wall_rel))

    return silence_hold(
        records, hop_seconds=HOP_SECONDS, clear_budget_s=clear_budget_s,
        out_dir=out_dir,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detector silence-hold / gap robustness A/B"
    )
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=96)
    args = ap.parse_args()
    out = build_silence_hold(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/silence_hold.json and .md")


if __name__ == "__main__":
    main()
