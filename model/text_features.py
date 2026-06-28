"""Shared text front-end: tokenize + vocab + vectorize (single source of truth).

This is the *golden* reference the Android/Kotlin side must mirror byte-for-byte.
The exported text ``.pte`` consumes the [1, V] float32 vector produced here; the
``.pte`` + the vocab asset are the contract, so this tokenizer/vectorizer pair is
frozen — any change here must be reflected in Kotlin or parity breaks.

Pipeline (must match Kotlin exactly):
  1. tokenize(text):
       t = text.lower()
       t = re.sub(r"[^a-z]+", " ", t)   # every non a-z char -> a single space
       return [w for w in t.split() if w]
  2. vectorize(tokens, vocab):
       counts = zeros(V)
       for tok in tokens: if tok in vocab: counts[vocab[tok]] += 1
       norm = sqrt(sum(counts**2)); vec = counts/norm if norm>0 else counts
     -> [1, V] float32 (L2-normalized term counts; all-zeros if no in-vocab token)

The vocab asset is one token per line, UTF-8, no header, no blank lines; line
index i (0-based) == feature index i.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# --- Vocab asset location (the contract file the Kotlin app loads) ---
VOCAB_PATH = Path("android/app/src/main/assets/text_stress_vocab.txt")

# Matches the Kotlin-side tokenizer exactly: collapse every run of non a-z chars
# (after lowercasing) into a separator, then split on whitespace.
_NON_ALPHA = re.compile(r"[^a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, replace every non a-z char with a space, split, drop empties.

    Unigrams only — no stemming, no stopword removal, no n-grams.
    """
    if text is None:
        return []
    t = text.lower()
    t = _NON_ALPHA.sub(" ", t)
    return [w for w in t.split() if w]


def load_vocab(path: str | Path = VOCAB_PATH) -> dict[str, int]:
    """Load the vocab asset into a ``token -> index`` map.

    One token per line; line index (0-based) is the feature index. No header,
    no blank lines. Returned dict preserves that index == line number.
    """
    path = Path(path)
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            token = line.rstrip("\n")
            if token:
                vocab[token] = i
    return vocab


def write_vocab(tokens: list[str], path: str | Path = VOCAB_PATH) -> None:
    """Write the vocab asset: one token per line, UTF-8, no header/blank lines.

    ``tokens`` is the ordered list whose position is the feature index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Join with newline and add a trailing newline; no blank lines, no header.
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def vectorize(tokens: list[str], vocab: dict[str, int]) -> torch.Tensor:
    """Tokens + vocab -> [1, V] float32 L2-normalized term-count vector.

    counts[idx] += 1 for each in-vocab token; then divide by the L2 norm (unless
    the norm is 0, in which case the all-zeros vector is returned unchanged).
    """
    V = len(vocab)
    counts = torch.zeros(V, dtype=torch.float32)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            counts[idx] += 1.0
    norm = torch.sqrt(torch.sum(counts * counts))
    if norm.item() > 0.0:
        counts = counts / norm
    return counts.reshape(1, V)


def text_to_vector(text: str, vocab: dict[str, int]) -> torch.Tensor:
    """Convenience: raw text -> [1, V] float32 vector (tokenize then vectorize)."""
    return vectorize(tokenize(text), vocab)
