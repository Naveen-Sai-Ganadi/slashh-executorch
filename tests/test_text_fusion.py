"""Parity + contract tests for the text-stress and fusion .pte path.

The exported text/fusion programs are only trustworthy if running them through
the ExecuTorch runtime reproduces the eager models within tolerance, and if the
golden vectors the Kotlin side asserts against match those programs. Loosening a
tolerance to force a pass is a bug.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from model.fusion import (
    FusionNet,
    example_input as fusion_example,
    export_to_pte as export_fusion,
    load_fusion_model,
)
from model.run_pte import run_pte
from model.text_features import load_vocab, tokenize, vectorize
from model.text_stress import (
    TextStressNet,
    example_input as text_example,
    export_to_pte as export_text,
    load_text_model,
)

MAX_ABS_ERR = 1e-3
GOLDEN = Path("model/golden_text_fusion.json")
TEXT_PTE = Path("android/app/src/main/assets/text_stress.pte")
FUSION_PTE = Path("android/app/src/main/assets/fusion.pte")
VOCAB = Path("android/app/src/main/assets/text_stress_vocab.txt")


def _export_tmp(buf: bytes, name: str) -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_bytes(buf)
    return path


def test_tokenize_contract():
    assert tokenize("WHY?!! Everything... is BROKEN -- 1234") == [
        "why", "everything", "is", "broken"
    ]
    assert tokenize("") == []
    assert tokenize("zzzqqq xxyy") == ["zzzqqq", "xxyy"]


def test_vectorize_is_l2_normalized():
    vocab = {"a": 0, "b": 1, "c": 2}
    vec = vectorize(["a", "a", "b"], vocab)
    assert vec.shape == (1, 3)
    assert abs(vec.norm().item() - 1.0) < 1e-6
    # all out-of-vocab / empty -> all zeros (norm 0, returned unchanged)
    assert vectorize(["zzz"], vocab).abs().sum().item() == 0.0


def test_vocab_asset_matches_model_dim():
    vocab = load_vocab(VOCAB)
    state = torch.load("assets/text_stress.pt", map_location="cpu")
    assert len(vocab) == state["vocab_size"]
    # contiguous 0..V-1 indices
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_text_pte_matches_eager():
    model = load_text_model()
    vocab_size = model.net[0].in_features
    x = text_example(vocab_size)
    with torch.no_grad():
        eager = model(x)
    pte = _export_tmp(export_text(model, vocab_size), "text.pte")
    out = run_pte(pte, x)
    assert out.shape == eager.shape
    assert (out - eager).abs().max().item() < MAX_ABS_ERR


def test_fusion_pte_matches_eager():
    model = load_fusion_model()
    x = fusion_example()
    with torch.no_grad():
        eager = model(x)
    pte = _export_tmp(export_fusion(model), "fusion.pte")
    out = run_pte(pte, x)
    assert out.shape == eager.shape
    assert (out - eager).abs().max().item() < MAX_ABS_ERR


def test_fusion_monotonic_and_confident():
    model = load_fusion_model()

    def f(a, t):
        with torch.no_grad():
            return float(model(torch.tensor([[a, t]], dtype=torch.float32)).item())

    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    # rising in text for fixed audio, and in audio for fixed text
    for a in levels:
        row = [f(a, t) for t in levels]
        assert all(row[i] <= row[i + 1] + 1e-6 for i in range(len(row) - 1))
    for t in levels:
        col = [f(a, t) for a in levels]
        assert all(col[i] <= col[i + 1] + 1e-6 for i in range(len(col) - 1))
    # agreement is more confident than either alone; disagreement hedges
    assert f(0.8, 0.8) > 0.8
    assert f(0.1, 0.1) < 0.1
    assert 0.0 < f(0.9, 0.1) < 1.0


def test_golden_matches_runtime():
    golden = json.loads(GOLDEN.read_text())
    vocab = load_vocab(VOCAB)
    assert golden["vocab_size"] == len(vocab)
    for case in golden["text_cases"]:
        vec = vectorize(tokenize(case["text"]), vocab)
        out = run_pte(TEXT_PTE, vec).item()
        assert abs(out - case["score"]) < MAX_ABS_ERR
    for case in golden["fusion_cases"]:
        inp = torch.tensor([[case["audio"], case["text"]]], dtype=torch.float32)
        out = run_pte(FUSION_PTE, inp).item()
        assert abs(out - case["score"]) < MAX_ABS_ERR
