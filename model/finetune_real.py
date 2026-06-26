"""Fine-tune StressNet on real emotional-speech corpora (RAVDESS + EMO-DB).

The shipped checkpoint (``assets/stress_model.pt``) is trained on the synthetic
proxy generator — enough to validate the train→export→run pipeline, but not real
voices. This module fine-tunes that same net on *real* labelled speech mapped to
the arousal axis (``model/realdata.py``), evaluated under a **speaker-independent
split** (no actor appears in both train and validation), and exports a new
checkpoint + ``.pte``.

Why speaker-independent: emotion classifiers readily latch onto *speaker
identity* instead of arousal, so a random clip-level split inflates accuracy.
Holding entire speakers out measures whether the model generalizes to unheard
voices — the only number worth trusting for on-device use.

    python -m model.finetune_real --ravdess <dir> --emodb <dir>

Writes ``docs/benchmarks/finetune_real.{json,md}``. With ``--out`` it also writes
the fine-tuned checkpoint and a ``.pte`` (overwriting shipped artifacts only when
you pass those paths explicitly). Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .model import StressNet, build_model
from .realdata import RealDataset, build_real_dataset


# --- speaker-independent split ---------------------------------------------

def speaker_split(
    speakers: list[str], *, val_fraction: float = 0.25, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Partition example indices so no speaker spans train and val.

    Speakers are held out *per corpus* (the prefix before ':') so both splits
    keep every corpus represented; otherwise a whole dataset could land entirely
    in train or val and the held-out number would mix arousal with domain shift.
    """
    by_corpus: dict[str, list[str]] = {}
    for s in speakers:
        by_corpus.setdefault(s.split(":")[0], []).append(s)

    gen = torch.Generator().manual_seed(seed)
    val_speakers: set[str] = set()
    for corpus, members in by_corpus.items():
        uniq = sorted(set(members))
        k = max(1, round(len(uniq) * val_fraction))
        perm = torch.randperm(len(uniq), generator=gen).tolist()
        val_speakers.update(uniq[i] for i in perm[:k])

    train_idx = [i for i, s in enumerate(speakers) if s not in val_speakers]
    val_idx = [i for i, s in enumerate(speakers) if s in val_speakers]
    return train_idx, val_idx


# --- metrics ----------------------------------------------------------------

@dataclass(frozen=True)
class EvalMetrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    per_corpus: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n, "accuracy": self.accuracy, "precision": self.precision,
            "recall": self.recall, "f1": self.f1,
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "per_corpus_accuracy": self.per_corpus,
        }


@torch.no_grad()
def _scores(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x).flatten()


def best_accuracy_threshold(
    scores: torch.Tensor, y: torch.Tensor, *, grid: int = 99
) -> float:
    """Pick the decision threshold maximizing accuracy on (scores, y).

    Selected on the *train* split only; the held-out report then applies it
    blind, so this is not test-set tuning.
    """
    yt = y.flatten()
    best_t, best_acc = 0.5, -1.0
    for i in range(1, grid + 1):
        t = i / (grid + 1)
        acc = float(((scores >= t).float() == yt).float().mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    speakers: list[str], *, threshold: float = 0.5,
) -> EvalMetrics:
    p = _scores(model, x)
    pred = (p >= threshold).float()
    yt = y.flatten()
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    acc = (tp + tn) / max(1, len(yt))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    per_corpus: dict[str, float] = {}
    corpora = sorted({s.split(":")[0] for s in speakers})
    for c in corpora:
        mask = torch.tensor([s.split(":")[0] == c for s in speakers])
        if mask.any():
            per_corpus[c] = float((pred[mask] == yt[mask]).float().mean())
    return EvalMetrics(len(yt), acc, prec, rec, f1, tp, fp, tn, fn, per_corpus)


# --- fine-tune --------------------------------------------------------------

@dataclass(frozen=True)
class FineTuneResult:
    train: EvalMetrics
    val: EvalMetrics
    val_speakers: list[str]
    train_n: int
    stressed_fraction: float
    init_from: str
    epochs: int
    lr: float
    tuned_threshold: float = 0.5
    val_tuned: EvalMetrics | None = None
    pte_bytes: int | None = None

    def to_dict(self) -> dict:
        return {
            "init_from": self.init_from,
            "epochs": self.epochs,
            "lr": self.lr,
            "train_n": self.train_n,
            "stressed_fraction": self.stressed_fraction,
            "val_speakers": self.val_speakers,
            "tuned_threshold": self.tuned_threshold,
            "train_metrics": self.train.to_dict(),
            "val_metrics": self.val.to_dict(),
            "val_metrics_tuned": self.val_tuned.to_dict() if self.val_tuned else None,
            "pte_bytes": self.pte_bytes,
        }


def _class_weights(y: torch.Tensor) -> torch.Tensor:
    """Per-sample inverse-frequency weights for class-balanced BCE."""
    pos = float(y.mean())
    neg = 1.0 - pos
    w_pos = 0.5 / max(pos, 1e-6)
    w_neg = 0.5 / max(neg, 1e-6)
    return torch.where(y > 0.5, torch.full_like(y, w_pos), torch.full_like(y, w_neg))


def finetune(
    dataset: RealDataset,
    *,
    init_weights: str | Path | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 5e-4,
    seed: int = 0,
    val_fraction: float = 0.25,
    threshold: float = 0.5,
) -> tuple[StressNet, FineTuneResult]:
    """Fine-tune StressNet on a RealDataset under a speaker-independent split."""
    torch.manual_seed(seed)
    train_idx, val_idx = speaker_split(
        dataset.speakers, val_fraction=val_fraction, seed=seed
    )
    xtr, ytr = dataset.x[train_idx], dataset.y[train_idx]
    xval, yval = dataset.x[val_idx], dataset.y[val_idx]
    sp_tr = [dataset.speakers[i] for i in train_idx]
    sp_val = [dataset.speakers[i] for i in val_idx]

    if init_weights is not None:
        model = build_model(str(init_weights))           # load shipped weights
        init_from = str(init_weights)
    else:
        model = StressNet()                              # default width (16,32,64)
        init_from = "random"
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(xtr.shape[0], generator=gen)
        for i in range(0, xtr.shape[0], batch_size):
            idx = order[i:i + batch_size]
            xb, yb = xtr[idx], ytr[idx]
            opt.zero_grad()
            p = model(xb)
            loss = nn.functional.binary_cross_entropy(
                p, yb, weight=_class_weights(yb)
            )
            loss.backward()
            opt.step()

    train_m = evaluate(model, xtr, ytr, sp_tr, threshold=threshold)
    val_m = evaluate(model, xval, yval, sp_val, threshold=threshold)
    # Pick the operating point on TRAIN scores, then apply it blind to held-out.
    tuned_t = best_accuracy_threshold(_scores(model, xtr), ytr)
    val_tuned = evaluate(model, xval, yval, sp_val, threshold=tuned_t)
    result = FineTuneResult(
        train=train_m, val=val_m,
        val_speakers=sorted(set(sp_val)),
        train_n=xtr.shape[0],
        stressed_fraction=dataset.stressed_fraction,
        init_from=init_from, epochs=epochs, lr=lr,
        tuned_threshold=tuned_t, val_tuned=val_tuned,
    )
    return model, result


def to_markdown(out: FineTuneResult) -> str:
    v, t = out.val, out.train
    pc = " · ".join(f"{k}: {a:.3f}" for k, a in v.per_corpus.items())
    md = (
        "# Fine-tune on real speech (RAVDESS + EMO-DB)\n\n"
        f"StressNet fine-tuned from **{out.init_from}** on real emotional speech "
        "mapped to the arousal axis, under a **speaker-independent** split "
        f"({out.train_n} train clips; {v.n} validation clips from held-out "
        f"speakers). Stressed (high-arousal) fraction of the pool: "
        f"**{out.stressed_fraction:.2f}**.\n\n"
        "Held-out (unseen speakers) is the number that matters:\n\n"
        f"- **accuracy: {v.accuracy:.3f}**\n"
        f"- precision: {v.precision:.3f} · recall: {v.recall:.3f} · "
        f"f1: {v.f1:.3f}\n"
        f"- confusion: tp={v.tp} fp={v.fp} tn={v.tn} fn={v.fn}\n"
        f"- per-corpus accuracy: {pc}\n\n"
    )
    if out.val_tuned is not None and abs(out.tuned_threshold - 0.5) > 1e-6:
        vt = out.val_tuned
        pct = " · ".join(f"{k}: {a:.3f}" for k, a in vt.per_corpus.items())
        md += (
            f"At 0.5 the model is conservative (high precision, low recall). "
            f"Selecting the operating point on the **train** split gives "
            f"threshold **{out.tuned_threshold:.2f}**; applied blind to held-out:\n\n"
            f"- **accuracy: {vt.accuracy:.3f}**\n"
            f"- precision: {vt.precision:.3f} · recall: {vt.recall:.3f} · "
            f"f1: {vt.f1:.3f}\n"
            f"- confusion: tp={vt.tp} fp={vt.fp} tn={vt.tn} fn={vt.fn}\n"
            f"- per-corpus accuracy: {pct}\n\n"
        )
    md += (
        f"Train-split accuracy (same speakers, for reference): {t.accuracy:.3f}\n\n"
        f"Held-out speakers: {', '.join(out.val_speakers)}\n"
    )
    if out.pte_bytes is not None:
        md += f"\nExported `.pte`: {out.pte_bytes / 1024:.1f} KB\n"
    return md


def build_finetune(
    *,
    ravdess: str | Path | None,
    emodb: str | Path | None,
    init_weights: str | Path | None = "assets/stress_model.pt",
    epochs: int = 40,
    seed: int = 0,
    out_dir: str | Path | None = "docs/benchmarks",
    ckpt_out: str | Path | None = None,
    pte_out: str | Path | None = None,
    limit: int | None = None,
) -> FineTuneResult:
    """Load corpora, fine-tune, evaluate held-out, record, optionally export."""
    specs: dict[str, str | Path] = {}
    if ravdess:
        specs["ravdess"] = ravdess
    if emodb:
        specs["emodb"] = emodb
    if not specs:
        raise ValueError("provide at least one of --ravdess / --emodb")

    dataset = build_real_dataset(specs, limit=limit)
    model, result = finetune(
        dataset, init_weights=init_weights, epochs=epochs, seed=seed
    )

    pte_bytes = None
    if pte_out is not None or ckpt_out is not None:
        if ckpt_out is not None:
            ckpt_out = Path(ckpt_out)
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(),
                 "meta": {"source": "finetune_real", **result.to_dict()}},
                ckpt_out,
            )
        if pte_out is not None:
            from .export_executorch import export_to_pte

            buf = export_to_pte(model=model)
            pte_bytes = len(buf)
            pte_out = Path(pte_out)
            pte_out.parent.mkdir(parents=True, exist_ok=True)
            pte_out.write_bytes(buf)
            result = FineTuneResult(**{**result.__dict__, "pte_bytes": pte_bytes})

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "finetune_real.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        (out_dir / "finetune_real.md").write_text(to_markdown(result))

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune StressNet on real speech")
    ap.add_argument("--ravdess", default=None, help="RAVDESS speech root dir")
    ap.add_argument("--emodb", default=None, help="EMO-DB root dir")
    ap.add_argument("--init", default="assets/stress_model.pt",
                    help="checkpoint to fine-tune from ('random' to train fresh)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="docs/benchmarks")
    ap.add_argument("--ckpt-out", default=None, help="write fine-tuned checkpoint here")
    ap.add_argument("--pte-out", default=None, help="also export a .pte here")
    ap.add_argument("--limit", type=int, default=None, help="cap clips (smoke test)")
    args = ap.parse_args()

    init = None if args.init == "random" else args.init
    out = build_finetune(
        ravdess=args.ravdess, emodb=args.emodb, init_weights=init,
        epochs=args.epochs, seed=args.seed, out_dir=args.out_dir,
        ckpt_out=args.ckpt_out, pte_out=args.pte_out, limit=args.limit,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/finetune_real.json and finetune_real.md")


if __name__ == "__main__":
    main()
