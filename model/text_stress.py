"""Text stress classifier: Dreaddit BoW -> tiny MLP -> ExecuTorch ``.pte``.

Run as a module:

    python -m model.text_stress

What it does (idempotent / re-runnable):
  1. Build a fixed vocabulary from the TRAIN split (document-frequency >= 2,
     top V=2000 by df with alphabetical tie-break) and write it to the Android
     asset ``text_stress_vocab.txt``.
  2. Vectorize train/test with the shared :mod:`model.text_features` front-end
     (L2-normalized term counts).
  3. Train a delegate-friendly MLP — ``Linear(V,64) -> ReLU -> Linear(64,1)`` with
     ``sigmoid`` in forward — class-balanced, weight decay, early stopping on an
     internal validation split. Report train/test accuracy + F1.
  4. Save the checkpoint to ``assets/text_stress.pt`` ({"model": ..., "vocab_size": V}).
  5. Export to ``android/app/src/main/assets/text_stress.pte`` via the same
     XNNPACK pattern as :func:`model.export_executorch.export_to_pte`.

The ``.pte`` consumes a [1, V] float32 vector; feature building lives outside the
graph (in :mod:`model.text_features`, mirrored on-device in Kotlin).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

from .export_executorch import _ensure_flatc
from .text_features import VOCAB_PATH, tokenize, vectorize, write_vocab

# --- Paths / config (the contract) ---
TRAIN_CSV = Path("data/dreaddit/dreaddit-train.csv")
TEST_CSV = Path("data/dreaddit/dreaddit-test.csv")
CHECKPOINT_PATH = Path("assets/text_stress.pt")
PTE_PATH = Path("android/app/src/main/assets/text_stress.pte")

MAX_VOCAB = 2000        # V (or fewer if not enough tokens clear df>=2)
MIN_DOC_FREQ = 2        # drop tokens appearing in < 2 training docs
SEED = 0


class TextStressNet(nn.Module):
    """BoW stress classifier: Linear(V,64) -> ReLU -> Linear(64,1) -> sigmoid.

    Only Linear/ReLU/Sigmoid so XNNPACK lowering has no fallback. Output [B,1]
    in [0,1] (1 == stressed).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def example_input(vocab_size: int, batch: int = 1) -> torch.Tensor:
    """Representative input for export / smoke tests: [batch, V] float32."""
    return torch.randn(batch, vocab_size, dtype=torch.float32)


def build_vocab(train_texts: list[str]) -> list[str]:
    """Document-frequency vocab from train texts.

    Count the number of train docs each token appears in (set per doc), drop
    df < MIN_DOC_FREQ, keep the top MAX_VOCAB by df with alphabetical tie-break
    for determinism. Returns the ordered token list (index == feature index).
    """
    doc_freq: Counter[str] = Counter()
    for text in train_texts:
        for token in set(tokenize(text)):
            doc_freq[token] += 1
    candidates = [(tok, c) for tok, c in doc_freq.items() if c >= MIN_DOC_FREQ]
    # df descending, then token ascending (alphabetical) for deterministic ties.
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return [tok for tok, _ in candidates[:MAX_VOCAB]]


def _vectorize_corpus(
    texts: list[str], vocab: dict[str, int]
) -> torch.Tensor:
    """Stack [1,V] vectors for every text into an [N, V] float32 matrix."""
    rows = [vectorize(tokenize(t), vocab) for t in texts]
    return torch.cat(rows, dim=0)


def _accuracy_f1(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        pred = (model(x) > 0.5).float()
    acc = (pred == y).float().mean().item()
    f1 = f1_score(y.numpy().ravel(), pred.numpy().ravel(), zero_division=0)
    return acc, float(f1)


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    vocab_size: int,
    *,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    patience: int = 12,
    seed: int = SEED,
) -> TextStressNet:
    """Class-balanced training with weight decay + early stopping on a val split.

    The test set is never used for selection — early stopping watches an internal
    validation slice carved from train. Returns the best-val model (eval mode).
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    n = x_train.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    xtr, ytr = x_train[tr_idx], y_train[tr_idx]
    xval, yval = x_train[val_idx], y_train[val_idx]

    model = TextStressNet(vocab_size)

    # Class-balanced: pos_weight = n_neg / n_pos on the training slice.
    n_pos = float(ytr.sum().item())
    n_neg = float(ytr.shape[0] - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for _ in range(epochs):
        model.train()
        batch_perm = torch.randperm(xtr.shape[0], generator=g)
        for i in range(0, xtr.shape[0], batch_size):
            idx = batch_perm[i:i + batch_size]
            opt.zero_grad()
            # BCEWithLogits wants logits: use the pre-sigmoid sub-net.
            logits = model.net(xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = (model(xval) > 0.5).float()
            val_acc = (val_pred == yval).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def export_to_pte(model: TextStressNet, vocab_size: int) -> bytes:
    """Serialize the text model to ``.pte`` bytes (XNNPACK fp32 delegate).

    Same flow as :func:`model.export_executorch.export_to_pte`: torch.export ->
    to_edge_transform_and_lower(XnnpackPartitioner) -> to_executorch.
    """
    _ensure_flatc()
    model.eval()
    example = (example_input(vocab_size),)
    exported = torch.export.export(model, example)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    return edge.to_executorch().buffer


def load_text_model(checkpoint: str | Path = CHECKPOINT_PATH) -> TextStressNet:
    """Rebuild the trained text model from its checkpoint (eval mode)."""
    state = torch.load(checkpoint, map_location="cpu")
    model = TextStressNet(state["vocab_size"])
    model.load_state_dict(state["model"])
    model.eval()
    return model


def run(verbose: bool = True) -> dict:
    """Build vocab, train, evaluate, save checkpoint + vocab asset + ``.pte``.

    Returns a metrics dict (vocab_size, train/test acc+f1) for the report.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_texts = train_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()
    y_train = torch.tensor(train_df["label"].values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_df["label"].values, dtype=torch.float32).unsqueeze(1)

    vocab_tokens = build_vocab(train_texts)
    vocab_size = len(vocab_tokens)
    write_vocab(vocab_tokens, VOCAB_PATH)
    vocab = {tok: i for i, tok in enumerate(vocab_tokens)}

    x_train = _vectorize_corpus(train_texts, vocab)
    x_test = _vectorize_corpus(test_texts, vocab)

    model = train_model(x_train, y_train, vocab_size)

    train_acc, train_f1 = _accuracy_f1(model, x_train, y_train)
    test_acc, test_f1 = _accuracy_f1(model, x_test, y_test)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": vocab_size}, CHECKPOINT_PATH)

    buffer = export_to_pte(model, vocab_size)
    PTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PTE_PATH.write_bytes(buffer)

    metrics = {
        "vocab_size": vocab_size,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "pte_bytes": len(buffer),
    }
    if verbose:
        print(f"vocab_size (V)   : {vocab_size}")
        print(f"vocab asset      : {VOCAB_PATH} ({len(vocab_tokens)} lines)")
        print(f"train acc / f1   : {train_acc:.4f} / {train_f1:.4f}")
        print(f"test  acc / f1   : {test_acc:.4f} / {test_f1:.4f}")
        print(f"checkpoint       : {CHECKPOINT_PATH}")
        print(f"pte              : {PTE_PATH} ({len(buffer):,} bytes)")
    return metrics


if __name__ == "__main__":
    run()
