"""S2/S3 deliverable — one comparison table across every model in the roster.

Aggregates the per-model benchmark JSONs produced by the training/export
drivers into a single leaderboard, so the 7-model roster (custom StressNet v2,
the frozen-backbone on-device encoders, and the SOTA teacher heads) is judged
on identical held-out-speaker balanced accuracy and, where exported, INT8
on-device parity.

Sources (all optional — rendered as they land):
  docs/benchmarks/train_unified_<tag>.json   custom-CNN training (held-out spk)
  docs/benchmarks/head_<backbone>_<map>.json frozen-backbone head training
  docs/benchmarks/export_<tag>.json           quantize+.pte parity

  python -m model.leaderboard

Writes docs/benchmarks/MODEL_LEADERBOARD.md. Pure file I/O — no model load,
no CPU contention with training/extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

BENCH = Path("docs/benchmarks")


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _rows():
    rows = []
    # custom CNN (and any v1/v2 unified-trained net)
    for p in sorted(BENCH.glob("train_unified_*.json")):
        d = _load(p)
        tag = p.stem.replace("train_unified_", "")
        exp = _load(BENCH / f"export_{tag}.json")
        rows.append({
            "model": tag,
            "kind": f"custom CNN ({d.get('arch','?')})",
            "mapping": d.get("mapping", "?"),
            "bal_acc": d.get("val_bal_acc"),
            "n_val": d.get("n_val"),
            "int8_bal_acc": exp.get("val_bal_acc_int8_pte"),
            "int8_agree": exp.get("agree_int8_vs_eager"),
            "int8_kb": (exp.get("int8_pte_bytes") or 0) / 1024 or None,
            "deployable": "on-device",
        })
    # frozen-backbone heads
    for p in sorted(BENCH.glob("head_*.json")):
        d = _load(p)
        bb = d.get("backbone", p.stem.replace("head_", ""))
        rows.append({
            "model": p.stem.replace("head_", ""),
            "kind": f"frozen {bb} + head",
            "mapping": d.get("mapping", "?"),
            "bal_acc": d.get("val_bal_acc"),
            "n_val": d.get("n_val"),
            "int8_bal_acc": None, "int8_agree": None, "int8_kb": None,
            "deployable": "encoder→head",
        })
    # LoRA-finetuned SOTA teachers
    for p in sorted(BENCH.glob("lora_*.json")):
        d = _load(p)
        bb = d.get("backbone", p.stem.replace("lora_", ""))
        tp = d.get("trainable_params")
        rows.append({
            "model": p.stem.replace("lora_", ""),
            "kind": (f"LoRA {bb}"
                     + (f" (r{d.get('lora_r','?')}, {tp/1e6:.1f}M tr)" if tp else "")),
            "mapping": d.get("mapping", "?"),
            "bal_acc": d.get("val_bal_acc"),
            "n_val": d.get("n_val"),
            "int8_bal_acc": None, "int8_agree": None, "int8_kb": None,
            "deployable": "teacher (cloud)",
        })
    return rows


def _fmt(v, spec="{:.3f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "—"


def render() -> str:
    rows = _rows()
    rows.sort(key=lambda r: (r["bal_acc"] is None, -(r["bal_acc"] or 0)))
    out = ["# Model leaderboard — held-out-speaker comparison", ""]
    out.append("Balanced accuracy on the unified speaker-independent val split. "
               "INT8 columns are on-device `.pte` parity (blank = not yet exported).")
    out.append("")
    out.append("| model | kind | map | val bal-acc | INT8 bal-acc | INT8 agree | INT8 KB |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| `{r['model']}` | {r['kind']} | {r['mapping']} | "
            f"{_fmt(r['bal_acc'])} | {_fmt(r['int8_bal_acc'])} | "
            f"{_fmt(r['int8_agree'])} | {_fmt(r['int8_kb'], '{:.0f}')} |"
        )
    if not rows:
        out.append("| _(no benchmarks yet)_ | | | | | | |")
    out.append("")
    done = [r for r in rows if r["bal_acc"] is not None]
    if done:
        best = max(done, key=lambda r: r["bal_acc"])
        out.append(f"**Best so far:** `{best['model']}` ({best['kind']}) "
                   f"at {best['bal_acc']:.3f} bal-acc.")
    return "\n".join(out) + "\n"


def main():
    BENCH.mkdir(parents=True, exist_ok=True)
    md = render()
    (BENCH / "MODEL_LEADERBOARD.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
