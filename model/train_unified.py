"""Train a stress classifier on the unified multi-corpus dataset.

Consumes ``datasets/unified/unified.pt`` (built by scratchpad/build_unified.py):
~15k real RAVDESS+CREMA-D+TESS+SAVEE log-mel windows with a *speaker-independent*
train/val split and two arousal label mappings (narrow / broad). This is the
honest training entry point for StressNet v2 and any head — it:

  * respects the precomputed speaker-independent ``split`` (no speaker leakage),
  * class-weights the BCE loss to counter the calm/stressed imbalance,
  * selects the checkpoint by held-out *balanced* accuracy, and
  * writes docs/benchmarks/train_unified_<tag>.{json,md}.

CLI:
  python -m model.train_unified --mapping broad --epochs 40 --tag stressnet_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .model import build_model, build_model_v2

UNIFIED = Path("datasets/unified/unified.pt")
BENCH = Path("docs/benchmarks")


def load_split(mapping: str):
    """Return (xtr, ytr, xval, yval, stats) for the chosen label mapping.

    ``mapping`` is 'narrow' (excludes ambiguous-arousal emotions, y==-1) or
    'broad' (Russell circumplex, all samples). Split comes from the unified
    artifact's speaker-independent ``split`` field.
    """
    d = torch.load(UNIFIED, weights_only=False)
    x = d["features"]                                   # [N,1,64,301]
    y = d["y_narrow" if mapping == "narrow" else "y_broad"]  # [N,1]
    split = d["split"]                                  # list 'train'/'val'

    is_val = torch.tensor([s == "val" for s in split])
    keep = (y.squeeze(1) >= 0)                          # drop y==-1 (narrow)
    tr_mask = (~is_val) & keep
    val_mask = is_val & keep

    xtr, ytr = x[tr_mask], y[tr_mask]
    xval, yval = x[val_mask], y[val_mask]
    stats = {
        "mapping": mapping,
        "n_train": int(tr_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_pos_frac": float(ytr.mean()),
        "val_pos_frac": float(yval.mean()),
    }
    return xtr, ytr, xval, yval, stats


def _balanced_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            preds.append(model(x[i:i + bs]))
    p = (torch.cat(preds) > 0.5).float()
    acc = (p == y).float().mean().item()
    per = []
    for c in (0.0, 1.0):
        m = (y == c)
        if m.any():
            per.append((p[m] == y[m]).float().mean().item())
    bal = sum(per) / len(per) if per else 0.0
    return acc, bal


def train_unified(
    *,
    mapping: str = "broad",
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    arch: str = "v2",
    device: str | None = None,
):
    torch.manual_seed(seed)
    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    xtr, ytr, xval, yval, stats = load_split(mapping)
    xtr, ytr = xtr.to(dev), ytr.to(dev)
    xval, yval = xval.to(dev), yval.to(dev)

    model = (build_model_v2() if arch == "v2" else build_model()).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # class-weighted BCE: weight the minority (stressed) class up so the loss
    # is balanced despite the ~calm-heavy mix. pos_weight = n_neg / n_pos.
    pos = float(ytr.sum())
    neg = float(ytr.numel() - ytr.sum())
    pos_w = neg / max(pos, 1.0)
    bce = nn.BCELoss(reduction="none")

    def weighted_loss(out, target):
        w = torch.where(target > 0.5, pos_w, 1.0)
        return (bce(out, target) * w).mean()

    gen = torch.Generator().manual_seed(seed)
    best_bal, best_state, best = 0.0, None, {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(xtr.shape[0], generator=gen)
        for i in range(0, xtr.shape[0], batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = weighted_loss(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        vacc, vbal = _balanced_acc(model, xval, yval)
        print(f"epoch {epoch:3d}/{epochs}  val_acc={vacc:.3f}  val_bal_acc={vbal:.3f}",
              flush=True)
        if vbal >= best_bal:
            best_bal = vbal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {"val_acc": vacc, "val_bal_acc": vbal, "epoch": epoch}

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {**stats, "arch": arch, "pos_weight": pos_w, "epochs": epochs,
            "lr": lr, "batch_size": batch_size, "seed": seed, **best}
    return model.cpu().eval(), best_state, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", choices=["narrow", "broad"], default="broad")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2")
    ap.add_argument("--tag", default="stressnet_v2")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, state, meta = train_unified(
        mapping=args.mapping, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, arch=args.arch, device=args.device,
    )

    ckpt_dir = Path("model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{args.tag}.pt"
    torch.save({"model": state, "meta": meta}, ckpt)

    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / f"train_unified_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    md = (
        f"# Unified training — {args.tag}\n\n"
        f"- arch: **{meta['arch']}**  mapping: **{meta['mapping']}**\n"
        f"- train/val: {meta['n_train']}/{meta['n_val']} "
        f"(speaker-independent; val pos frac {meta['val_pos_frac']:.3f})\n"
        f"- pos_weight: {meta['pos_weight']:.3f}\n"
        f"- best epoch {meta.get('epoch')}: "
        f"val_acc **{meta.get('val_acc', 0):.3f}**, "
        f"val_bal_acc **{meta.get('val_bal_acc', 0):.3f}**\n"
    )
    (BENCH / f"train_unified_{args.tag}.md").write_text(md)
    print(f"\nsaved {ckpt}\nval_bal_acc={meta.get('val_bal_acc', 0):.3f}  "
          f"(held-out speakers, {meta['mapping']} mapping)")


if __name__ == "__main__":
    main()
