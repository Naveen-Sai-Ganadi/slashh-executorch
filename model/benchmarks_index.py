"""Roll the docs/benchmarks/*.json artifacts into one README.

The autonomy loop emits several JSON records under ``docs/benchmarks/`` —
host benchmarks, the trained width A/B, detector tuning, the noise-robustness
sweep, and the augmented-training comparison. This generator summarizes
whichever are present into a single index so a human waking up to the night's
work (or the next loop iteration) has one legible entry point.

    python -m model.benchmarks_index            # writes docs/benchmarks/README.md

Defensive by design: only existing artifacts are included, malformed JSON is
skipped silently, and an empty directory yields a placeholder rather than an
error. Host-only; reads nothing but the local artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _fmt_floor(v) -> str:
    return "none (holds at all tested SNRs)" if v is None else f"{v:g} dB"


def _summarize_benchmark(d: dict) -> str:
    rows = d.get("results", [])
    if not rows:
        return "_(no rows)_"
    parts = []
    for r in rows:
        name = r.get("name", "?")
        kb = r.get("pte_bytes", 0) / 1024
        lat = r.get("pte_latency_ms")
        lat_s = f", {lat:.3f} ms" if isinstance(lat, (int, float)) else ""
        parts.append(f"`{name}` ({kb:.1f} KB{lat_s})")
    return "Variants: " + ", ".join(parts) + "."


def _summarize_ab(d: dict) -> str:
    rec = d.get("recommended", "?")
    n = len(d.get("results", []))
    return f"Trained {n} width variant(s); recommended **`{rec}`**."


def _summarize_tuning(d: dict) -> str:
    b = d.get("best", {})
    return (
        f"Best knobs: stress={b.get('stress_threshold')}, "
        f"release={b.get('release_threshold')}, alpha={b.get('ema_alpha')} "
        f"(acc {b.get('accuracy')}, flicker {b.get('flicker')})."
    )


def _summarize_robustness(d: dict) -> str:
    return f"Operating floor: **{_fmt_floor(d.get('floor_db'))}**."


def _summarize_production(d: dict) -> str:
    ch = tuple(d.get("channels", []))
    kb = d.get("pte_bytes", 0) / 1024
    rob = d.get("robustness", {})
    reliable = rob.get("reliable_floor_db")
    if reliable is not None:
        floor_txt = f"reliable to {reliable:g} dB"
    else:
        # older records (no reliable_floor_db) fall back to the failing floor
        floor_txt = f"floor {_fmt_floor(rob.get('floor_db'))}"
    msg = (
        f"Shipped width **`{ch}`** ({d.get('params', '?'):,} params, {kb:.1f} KB "
        f".pte), clean acc {d.get('val_acc', float('nan')):.3f}, "
        f"{floor_txt}."
    )
    int8 = d.get("int8_bytes")
    if int8 is not None:
        msg += (
            f" INT8 variant {int8 / 1024:.1f} KB "
            f"(within {d.get('int8_max_abs_diff', float('nan')):.4f} of eager)."
        )
    return msg


def _summarize_robust_train(d: dict) -> str:
    base = d.get("baseline", {}).get("floor_db")
    aug = d.get("augmented", {}).get("floor_db")
    return (
        f"Clean-trained floor {_fmt_floor(base)} → augmented floor "
        f"**{_fmt_floor(aug)}**."
    )


# (filename, human title, summarizer). Order = display order.
_ARTIFACTS = [
    ("benchmark.json", "Host benchmarks (XNNPACK-CPU)", _summarize_benchmark),
    ("ab_experiment.json", "Trained architecture A/B", _summarize_ab),
    ("detector_tuning.json", "Detector tuning sweep", _summarize_tuning),
    ("robustness.json", "Noise robustness", _summarize_robustness),
    ("robust_train.json", "Noise-augmented training", _summarize_robust_train),
    ("production.json", "Production model (shipped recipe)", _summarize_production),
]


def build_index(bench_dir: str | Path, *, write: bool = True) -> str:
    """Summarize present artifacts in ``bench_dir`` into markdown.

    Writes ``README.md`` into the directory unless ``write=False``. Missing or
    malformed artifacts are skipped. Returns the markdown string.
    """
    bench_dir = Path(bench_dir)
    sections: list[str] = []
    for filename, title, summarize in _ARTIFACTS:
        path = bench_dir / filename
        if not path.is_file():
            continue
        data = _load(path)
        if not isinstance(data, dict):
            continue  # malformed — skip silently
        try:
            blurb = summarize(data)
        except Exception:
            blurb = "_(could not summarize)_"
        md_name = filename.replace(".json", ".md")
        link = f" · [details]({md_name})" if (bench_dir / md_name).is_file() else ""
        sections.append(f"### {title}\n\n{blurb} _(`{filename}`{link})_\n")

    header = (
        "# StressNet benchmarks & experiments\n\n"
        "_Auto-generated index of the artifacts in this directory "
        "(`python -m model.benchmarks_index`). Each entry links to its full "
        "table where available._\n\n"
    )
    if sections:
        body = "\n".join(sections)
    else:
        body = "_No benchmark artifacts found yet — run the harnesses in `model/`._\n"

    md = header + body

    if write:
        bench_dir.mkdir(parents=True, exist_ok=True)
        (bench_dir / "README.md").write_text(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the benchmarks README index")
    ap.add_argument("--bench-dir", default="docs/benchmarks")
    args = ap.parse_args()
    build_index(args.bench_dir)
    print(f"wrote {args.bench_dir}/README.md")


if __name__ == "__main__":
    main()
