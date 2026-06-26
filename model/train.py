"""Train StressNet and save a checkpoint the exporter can load.

    # synthetic smoke-train (offline, ~seconds) → checkpoint + exported .pte
    python -m model.train --epochs 8 --out assets/stress_model.pt

    # real data once it's on disk (plan §13 mapping → calm/ stressed/)
    python -m model.train --data-dir data/arousal --epochs 40

The checkpoint is ``{"model": state_dict, "meta": {...}}`` — exactly what
``build_model(weights)`` and ``export_to_pte(weights=...)`` expect. Training is
deliberately small and CPU-only: the point of Phase 0 is a *correct, exportable*
pipeline, not accuracy. Real datasets slot in via ``--data-dir`` with no code
change (see model/data.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .data import folder_dataset, ravdess_dataset, synthetic_dataset
from .model import build_model

# Default held-out RAVDESS actors for the speaker-independent split (unseen voices).
RAVDESS_VAL_ACTORS = (21, 22, 23, 24)


def _split(x: torch.Tensor, y: torch.Tensor, val_frac: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=gen)
    n_val = max(1, int(x.shape[0] * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


@torch.no_grad()
def _metrics(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float, float]:
    model.eval()
    p = model(x)
    loss = nn.functional.binary_cross_entropy(p, y).item()
    pred = (p >= 0.5).float()
    acc = (pred == y).float().mean().item()
    # balanced accuracy — robust to class imbalance
    per = [(pred[y == c] == c).float().mean().item()
           for c in (0.0, 1.0) if (y == c).any()]
    bal = sum(per) / len(per)
    return loss, acc, bal


def train(
    *,
    data_dir: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    n_per_class: int,
    seed: int,
    ravdess: str | None = None,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)

    if ravdess:
        # Real RAVDESS with a SPEAKER-INDEPENDENT split (honest accuracy).
        x, y, actors = ravdess_dataset(ravdess)
        val_mask = torch.tensor([int(a) in RAVDESS_VAL_ACTORS for a in actors])
        xtr, ytr = x[~val_mask], y[~val_mask]
        xval, yval = x[val_mask], y[val_mask]
        source = f"ravdess:{ravdess} (speaker-independent, val_actors={RAVDESS_VAL_ACTORS})"
    elif data_dir:
        x, y = folder_dataset(data_dir)
        xtr, ytr, xval, yval = _split(x, y, val_frac=0.2, seed=seed)
        source = f"folder:{data_dir}"
    else:
        x, y = synthetic_dataset(n_per_class, seed=seed)
        xtr, ytr, xval, yval = _split(x, y, val_frac=0.2, seed=seed)
        source = "synthetic"

    model = build_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_bal, best_state, best = 0.0, None, {}
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(xtr.shape[0], generator=gen)
        for i in range(0, xtr.shape[0], batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        vloss, vacc, vbal = _metrics(model, xval, yval)
        print(f"epoch {epoch:3d}/{epochs}  val_loss={vloss:.4f}  "
              f"val_acc={vacc:.3f}  val_bal_acc={vbal:.3f}")
        if vbal >= best_bal:
            best_bal = vbal
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best = {"val_loss": vloss, "val_acc": vacc, "val_bal_acc": vbal}

    # restore the best checkpoint (val accuracy can wobble across epochs)
    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "source": source,
        "n_train": int(xtr.shape[0]),
        "n_val": int(xval.shape[0]),
        "val_loss": best.get("val_loss"),
        "val_acc": best.get("val_acc"),
        "val_bal_acc": best.get("val_bal_acc"),
        "epochs": epochs,
        "seed": seed,
    }
    return model, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Train StressNet")
    ap.add_argument("--data-dir", default=None,
                    help="root with calm/ and stressed/ wavs; omit for synthetic")
    ap.add_argument("--ravdess", default=None,
                    help="RAVDESS root (Actor_*/ *.wav); speaker-independent split")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-per-class", type=int, default=64,
                    help="synthetic samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/stress_model.pt",
                    help="checkpoint output path")
    ap.add_argument("--export", default=None,
                    help="also export a .pte to this path after training")
    args = ap.parse_args()

    model, meta = train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_per_class=args.n_per_class,
        seed=args.seed,
        ravdess=args.ravdess,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, out)
    print(f"saved checkpoint {out}  (val_acc={meta['val_acc']:.3f}  "
          f"val_bal_acc={meta['val_bal_acc']:.3f}  source={meta['source']})")

    if args.export:
        from .export_executorch import export_to_pte

        buf = export_to_pte(model=model)
        pte = Path(args.export)
        pte.parent.mkdir(parents=True, exist_ok=True)
        pte.write_bytes(buf)
        print(f"exported {pte}  ({len(buf):,} bytes)")


if __name__ == "__main__":
    main()
