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


def _summarize_int8_robustness(d: dict) -> str:
    fp = d.get("fp32_reliable_floor_db")
    q = d.get("int8_reliable_floor_db")
    verdict = "preserves the floor" if d.get("preserves_floor") else "degrades the floor"
    return (
        f"INT8 vs fp32 under noise: fp32 reliable to {_fmt_floor(fp)}, "
        f"INT8 to {_fmt_floor(q)} — **INT8 {verdict}**."
    )


def _summarize_int8_calib_ab(d: dict) -> str:
    verdict = d.get("verdict", "?")
    rec = d.get("recommended", "?")
    by = {r.get("label"): r for r in d.get("results", [])}
    cf = by.get("clean", {}).get("reliable_floor_db")
    nf = by.get("noise-aware", {}).get("reliable_floor_db")
    gloss = {
        "better": "noise-aware calibration **deepens the floor**",
        "same": "**no difference** — clean calibration is sufficient",
        "worse": "noise-aware calibration is **worse**",
    }.get(verdict, verdict)
    return (
        f"INT8 calibration A/B (clean vs noise-aware): clean reliable to "
        f"{_fmt_floor(cf)}, noise-aware to {_fmt_floor(nf)} — {gloss}; "
        f"ship **`{rec}`** calibration."
    )


def _summarize_latency_rtf(d: dict) -> str:
    results = d.get("results", [])
    if not results:
        return "_(no backends timed)_"
    fastest = d.get("fastest", "?")
    best = min(results, key=lambda r: r.get("total_mean_ms", float("inf")))
    rtf_hop = best.get("rtf_hop")
    headroom = (
        f", ~{1.0 / rtf_hop:.0f}× real-time head-room" if rtf_hop else ""
    )
    rt = "**all backends real-time**" if d.get("real_time_all") else "**a backend misses the hop budget**"
    return (
        f"End-to-end PCM→log-mel→score: fastest **`{fastest}`** at "
        f"{best.get('total_mean_ms', float('nan')):.3f} ms/window (hop "
        f"{d.get('hop_seconds')}s{headroom}) — {rt}."
    )


def _summarize_calibration(d: dict) -> str:
    verdict = d.get("verdict", "?")
    ece = d.get("ece")
    t = d.get("temperature")
    after = d.get("ece_after_temp")
    gloss = {
        "well-calibrated": "scores are trustworthy probabilities — the gate is sound",
        "overconfident": "scores claim more certainty than earned",
        "underconfident": "scores under-state certainty — the gate is conservative",
    }.get(verdict, verdict)
    temp_txt = (
        f"; temperature T={t:g} → ECE {after:.3f}"
        if isinstance(t, (int, float)) and t != 1.0 else ""
    )
    ece_txt = f"{ece:.3f}" if isinstance(ece, (int, float)) else "?"
    return f"Score calibration: **{verdict}** (ECE {ece_txt}){temp_txt} — {gloss}."


def _summarize_calibration_envelope(d: dict) -> str:
    n = d.get("n_inits", "?")
    lo, hi = d.get("temp_min"), d.get("temp_max")
    med = d.get("temp_median")
    default = d.get("default_temperature")
    verdict = d.get("verdict", "?")
    span = (
        f"{lo:.3f}…{hi:.3f} (median {med:.3f})"
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float))
        else "?"
    )
    dtxt = f"{default:g}" if isinstance(default, (int, float)) else "?"
    return (
        f"Calibration-temperature envelope across {n} inits: T in {span} vs shipped "
        f"`DEFAULT_TEMPERATURE`={dtxt} — {verdict}."
    )


def _summarize_calibration_snr(d: dict) -> str:
    n = d.get("n_levels", "?")
    floor = d.get("floor_snr_db")
    floor_label = "clean" if floor is None else f"{floor:g} dB"
    trust = d.get("floor_trustworthy")
    grows = d.get("ece_increases_with_noise")
    return (
        f"Per-SNR calibration across {n} noise levels: ECE grows with noise: "
        f"{grows}; the confidence read-out is trustworthy at the {floor_label} "
        f"floor: {trust}."
    )


def _summarize_int8_calibration_drift(d: dict) -> str:
    fp32 = d.get("fp32", {})
    int8 = d.get("int8", {})
    drift = d.get("ece_drift")
    transfers = d.get("temperature_transfers")
    f_ece, q_ece = fp32.get("ece"), int8.get("ece")
    span = (
        f"fp32 {f_ece:.4f} -> INT8 {q_ece:.4f} (drift {drift:+.4f})"
        if all(isinstance(v, (int, float)) for v in (f_ece, q_ece, drift))
        else "?"
    )
    return (
        f"fp32->INT8 confidence-calibration drift over {d.get('n', '?')} windows: "
        f"ECE {span}; shipped fp32 temperature transfers to the deployed INT8 "
        f"model: {transfers}."
    )


def _summarize_snr_aware_temperature_ab(d: dict) -> str:
    pg = d.get("pooled_ece_global")
    po = d.get("pooled_ece_oracle")
    red = d.get("pooled_reduction")
    worth = d.get("worth_it")
    degen = d.get("oracle_degenerate")
    floor = d.get("floor_snr_db")
    floor_red = d.get("floor_reduction")
    floor_label = "clean" if floor is None else f"{floor:g} dB"
    span = (
        f"pooled ECE {pg:.4f} -> {po:.4f} (recovers {red:+.4f}), floor "
        f"{floor_red:+.4f} at {floor_label}"
        if all(isinstance(v, (int, float)) for v in (pg, po, red, floor_red))
        else "?"
    )
    return (
        f"global vs unconstrained-oracle SNR-aware temperature over "
        f"{d.get('n_levels', '?')} noise levels: {span}; oracle relies on "
        f"degenerate sub-floor T: {degen}; worth a deployable SNR estimator: "
        f"{worth}."
    )


def _summarize_feature_batching(d: dict) -> str:
    sp = d.get("speedup")
    n = d.get("n")
    ok = d.get("parity_ok")
    sp_txt = f"{sp:.2f}×" if isinstance(sp, (int, float)) else "?"
    parity = "parity holds" if ok else "**parity BREAK**"
    return (
        f"Batched log-mel front-end (`extract_batch`) is **{sp_txt}** faster "
        f"than the per-sample loop building {n} windows ({parity}) — host "
        "throughput for dataset/A-B builds; device extractor unchanged."
    )


def _summarize_noise_colors(d: dict) -> str:
    floors = d.get("reliable_floor_db", {})
    parts = ", ".join(f"{c} {_fmt_floor(v)}" for c, v in floors.items())
    verdict = (
        "holds across colors" if d.get("holds_across_colors") else "breaks for some color"
    )
    return f"Reliable floor by noise color: {parts} — **{verdict}**."


def _summarize_noise_failure_mode(d: dict) -> str:
    dom = d.get("dominant_failure")
    if dom is None:
        return "No dominant failure direction — holds above threshold at every tested SNR."
    gloss = {
        "misses_stress": "misses stress (false negatives — detector goes silent)",
        "false_alarms": "false alarms (false positives — detector cries wolf)",
        "balanced": "errors balanced (no dominant direction)",
    }.get(dom, dom)
    return f"First failure under noise leans **{gloss}**."


def _summarize_init_envelope(d: dict) -> str:
    env = d.get("envelope_db")
    n = d.get("n_inits", "?")
    if env is None:
        return (
            f"Across {n} inits, no common noisy envelope — some init is reliable "
            "only on clean audio."
        )
    return (
        f"Across {n} independent inits, the conservative envelope is reliable "
        f"down to **{_fmt_floor(env)}** (holds regardless of training seed)."
    )


def _summarize_detector_robustness(d: dict) -> str:
    floor = d.get("detection_floor_db")
    wc = d.get("window_count", "?")
    if floor is None:
        return (
            f"End-to-end detector ({wc}-window traces): **no noisy detection "
            "floor** — reliable only on clean audio."
        )
    return (
        f"End-to-end detector ({wc}-window traces): latches stress reliably "
        f"down to **{_fmt_floor(floor)}** (detect target met, false alarms in "
        "tolerance)."
    )


def _summarize_detector_noise_ab(d: dict) -> str:
    rec = d.get("recommended")
    tol = d.get("fa_tolerance")
    tol_txt = f"{tol:.0%}" if isinstance(tol, (int, float)) else "budget"
    if not rec:
        return (
            f"Detector knobs A/B'd under noise: **no config** stays within the "
            f"{tol_txt} false-alarm budget across the sweep — keep the default "
            "and treat noisy regimes as low-confidence."
        )
    cfg = rec.get("config", {})
    floor = rec.get("detection_floor_db")
    return (
        f"Detector knobs A/B'd under noise: recommend **`{cfg.get('label')}`** "
        f"(stress={cfg.get('stress_threshold')}, alpha={cfg.get('ema_alpha')}) "
        f"— detection floor {_fmt_floor(floor)}, worst false alarm "
        f"{rec.get('worst_false_alarm_rate')} (≤ {tol_txt})."
    )


def _summarize_augmentation_ab(d: dict) -> str:
    rec = d.get("recommended")
    n = len(d.get("results", []))
    if not rec:
        floor_txt = d.get("min_clean_acc")
        return (
            f"A/B'd {n} augmentation recipe(s): **none** kept clean accuracy "
            f"above the bar — loosen augmentation or the clean-accuracy floor."
        )
    label = rec.get("recipe", {}).get("label")
    floor = rec.get("reliable_floor_db")
    return (
        f"A/B'd {n} augmentation recipe(s): recommend **`{label}`** — reliable "
        f"floor {_fmt_floor(floor)}, clean acc {rec.get('clean_acc')}."
    )


def _summarize_recipe_envelope(d: dict) -> str:
    dee = d.get("deepest")
    n = d.get("n_inits", "?")
    envs = d.get("envelopes", [])
    if not dee:
        return (
            f"Recipe envelopes across {n} inits: **no recipe yields a common "
            "noisy envelope** — some init holds only on clean audio."
        )
    label = dee.get("recipe", {}).get("label")
    env = dee.get("envelope_db")
    # Did the deepest envelope actually beat the others, or tie?
    others = [e.get("envelope_db") for e in envs if e is not dee
              and e.get("recipe", {}).get("label") != label]
    real = [v for v in others if v is not None]
    moved = real and env is not None and env < max(real)
    verdict = "moves the envelope" if moved else "ties the envelope (single-seed gain was init-sensitive)"
    return (
        f"Recipe envelopes across {n} inits: deepest is **`{label}`** reliable "
        f"to {_fmt_floor(env)} — **{verdict}**."
    )


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
    ("int8_robustness.json", "INT8 vs fp32 robustness", _summarize_int8_robustness),
    ("int8_calib_ab.json", "INT8 calibration A/B (clean vs noise-aware)", _summarize_int8_calib_ab),
    ("latency_rtf.json", "End-to-end latency & Real-Time Factor", _summarize_latency_rtf),
    ("calibration.json", "Score calibration (reliability & ECE)", _summarize_calibration),
    ("calibration_envelope.json", "Cross-init calibration-temperature envelope", _summarize_calibration_envelope),
    ("int8_calibration_drift.json", "fp32 -> INT8 calibration drift", _summarize_int8_calibration_drift),
    ("calibration_snr.json", "Per-SNR calibration breakdown", _summarize_calibration_snr),
    ("snr_aware_temperature_ab.json", "SNR-aware vs global temperature (A/B)", _summarize_snr_aware_temperature_ab),
    ("feature_batching.json", "Front-end throughput (loop vs batched)", _summarize_feature_batching),
    ("noise_colors.json", "Robustness across noise colors", _summarize_noise_colors),
    ("noise_failure_mode.json", "Failure mode under noise", _summarize_noise_failure_mode),
    ("init_envelope.json", "Cross-initialization envelope", _summarize_init_envelope),
    ("detector_robustness.json", "End-to-end detector robustness", _summarize_detector_robustness),
    ("detector_noise_ab.json", "Detector A/B under noise", _summarize_detector_noise_ab),
    ("robust_train.json", "Noise-augmented training", _summarize_robust_train),
    ("augmentation_ab.json", "Augmentation-recipe A/B", _summarize_augmentation_ab),
    ("recipe_envelope.json", "Recipe-parameterized cross-init envelope", _summarize_recipe_envelope),
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
