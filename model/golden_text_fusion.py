"""Emit golden parity vectors for the Kotlin unit tests.

Run as a module (after model.text_stress and model.fusion have produced the
checkpoints + vocab asset):

    python -m model.golden_text_fusion

Writes ``model/golden_text_fusion.json`` with:
  - vocab_size V
  - text_cases: per case the raw text, its tokens, the nonzero feature entries
    {index: value} of the L2-normalized vector, and the eager text-model score.
  - fusion_cases: the full 5x5 grid of eager fusion-model outputs over
    audio,text in {0.0,0.25,0.5,0.75,1.0}.

All scores are the eager PyTorch outputs of the trained models, rounded to 6
decimals. These are the source of truth the Kotlin side asserts against.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .text_features import load_vocab, tokenize, vectorize
from .text_stress import load_text_model
from .fusion import load_fusion_model

OUTPUT_PATH = Path("model/golden_text_fusion.json")

# ~6 cases: clearly-stressed, calm, empty, all-out-of-vocab, mixed,
# punctuation/caps (to exercise tokenization).
TEXT_CASES = [
    "I am so overwhelmed and anxious, I can't sleep and everything feels like it's falling apart.",
    "We had a relaxing afternoon in the garden and enjoyed a calm, peaceful walk together.",
    "",
    "zzzqqq xxyy",
    "Work was fine but I keep worrying about the deadline and feel a bit stressed lately.",
    "WHY?!! Everything... is BROKEN -- 1234 #@$ totally Overwhelmed!!!",
]

FUSION_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _round6(x: float) -> float:
    return round(float(x), 6)


def build_golden() -> dict:
    vocab = load_vocab()
    vocab_size = len(vocab)
    text_model = load_text_model()
    fusion_model = load_fusion_model()

    text_cases = []
    for text in TEXT_CASES:
        tokens = tokenize(text)
        vec = vectorize(tokens, vocab)  # [1, V]
        flat = vec.squeeze(0)
        nonzero = {
            str(int(i)): _round6(flat[i].item())
            for i in torch.nonzero(flat, as_tuple=False).flatten().tolist()
        }
        with torch.no_grad():
            score = float(text_model(vec).item())
        text_cases.append({
            "text": text,
            "tokens": tokens,
            "nonzero": nonzero,
            "score": _round6(score),
        })

    fusion_cases = []
    with torch.no_grad():
        for a in FUSION_LEVELS:
            for t in FUSION_LEVELS:
                inp = torch.tensor([[a, t]], dtype=torch.float32)
                score = float(fusion_model(inp).item())
                fusion_cases.append({
                    "audio": a,
                    "text": t,
                    "score": _round6(score),
                })

    return {
        "vocab_size": vocab_size,
        "text_cases": text_cases,
        "fusion_cases": fusion_cases,
    }


def run(verbose: bool = True) -> dict:
    golden = build_golden()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    if verbose:
        print(f"wrote {OUTPUT_PATH}  (V={golden['vocab_size']}, "
              f"{len(golden['text_cases'])} text cases, "
              f"{len(golden['fusion_cases'])} fusion cases)")
        for c in golden["text_cases"]:
            preview = (c["text"][:40] + "...") if len(c["text"]) > 40 else c["text"]
            print(f"  text[{len(c['tokens'])} tok] score={c['score']:.6f}  {preview!r}")
    return golden


if __name__ == "__main__":
    run()
