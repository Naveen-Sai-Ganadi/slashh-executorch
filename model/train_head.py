"""Train a lightweight stress head on cached frozen-backbone embeddings.

Consumes ``datasets/unified/emb_<backbone>.pt`` (built by
scratchpad/extract_embeddings.py): pooled embeddings for whisper-tiny / wavlm /
hubert / audeering aligned to the unified speaker-independent split. Trains a
small MLP head (the only part that ships per-backbone for the frozen-backbone
models) with the same class-weighted BCE + balanced-accuracy selection used for
StressNet, so every model is evaluated on identical held-out speakers.

CLI:
  python -m model.train_head --backbone whisper --mapping broad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

EMB_DIR = Path("datasets/unified")
BENCH = Path("docs/benchmarks")


class Head(nn.Module):
    """emb -> stress score in [0,1]. Small, quantizes cleanly, trivially export."""

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load(backbone: str, mapping: str):
    d = torch.load(EMB_DIR / f"emb_{backbone}.pt", weights_only=False)
    emb = d["emb"].float()
    y = d["y_narrow" if mapping == "narrow" else "y_broad"].float()
    is_val = torch.tensor([s == "val" for s in d["split"]])
    keep = (y.squeeze(1) >= 0)
    tr = (~is_val) & keep
    va = is_val & keep
    return emb[tr], y[tr], emb[va], y[va], emb.shape[1]


def _bal(model, x, y):
    model.eval()
    with torch.no_grad():
        p = (model(x) > 0.5).float()
    per = []
    for c in (0.0, 1.0):
        m = (y == c)
        if m.any():
            per.append((p[m] == y[m]).float().mean().item())
    acc = (p == y).float().mean().item()
    return acc, (sum(per) / len(per) if per else 0.0)


def train_head(backbone: str, mapping: str = "broad", epochs: int = 60,
               lr: float = 1e-3, batch_size: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    xtr, ytr, xva, yva, dim = load(backbone, mapping)
    # standardize embeddings (frozen features have arbitrary scale)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd

    model = Head(dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")
    pos_w = float((ytr.numel() - ytr.sum()) / max(float(ytr.sum()), 1.0))

    gen = torch.Generator().manual_seed(seed)
    best_bal, best_state, best = 0.0, None, {}
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(xtr.shape[0], generator=gen)
        for i in range(0, xtr.shape[0], batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            out = model(xtr[idx])
            w = torch.where(ytr[idx] > 0.5, pos_w, 1.0)
            (bce(out, ytr[idx]) * w).mean().backward()
            opt.step()
        acc, bal = _bal(model, xva, yva)
        if bal >= best_bal:
            best_bal, best = bal, {"val_acc": acc, "val_bal_acc": bal, "epoch": ep}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    meta = {"backbone": backbone, "mapping": mapping, "emb_dim": dim,
            "n_train": int(xtr.shape[0]), "n_val": int(xva.shape[0]),
            "pos_weight": pos_w, "epochs": epochs, **best}
    return model, best_state, (mu, sd), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=["whisper", "wavlm", "hubert", "audeering"])
    ap.add_argument("--mapping", choices=["narrow", "broad"], default="broad")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    model, state, (mu, sd), meta = train_head(args.backbone, args.mapping, args.epochs)
    tag = f"{args.backbone}_{args.mapping}"
    ckpt_dir = Path("model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "norm": {"mu": mu, "sd": sd}, "meta": meta},
               ckpt_dir / f"head_{tag}.pt")

    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / f"head_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{tag}] val_acc={meta.get('val_acc', 0):.3f}  "
          f"val_bal_acc={meta.get('val_bal_acc', 0):.3f}  (dim={meta['emb_dim']})")


if __name__ == "__main__":
    main()
