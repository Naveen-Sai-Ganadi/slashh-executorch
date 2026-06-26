"""Evaluate a checkpoint, and (optionally) the exported ``.pte``.

    python -m model.eval --weights assets/stress_model.pt
    python -m model.eval --weights assets/stress_model.pt --pte assets/stress_model.pte

Reports accuracy / precision / recall / a small confusion matrix on a held-out
synthetic split (or a real ``--data-dir``). With ``--pte`` it also checks that
the exported program agrees with eager PyTorch on the eval set — a fleet-wide
parity check beyond the single-input gate in tests/test_parity.py.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from .data import folder_dataset, synthetic_dataset
from .model import build_model


def _confusion(pred: torch.Tensor, y: torch.Tensor) -> dict:
    pred_b = (pred >= 0.5).float()
    tp = int(((pred_b == 1) & (y == 1)).sum())
    tn = int(((pred_b == 0) & (y == 0)).sum())
    fp = int(((pred_b == 1) & (y == 0)).sum())
    fn = int(((pred_b == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
    }


def evaluate(weights: str, *, data_dir: str | None, pte: str | None, seed: int) -> dict:
    if data_dir:
        x, y = folder_dataset(data_dir)
    else:
        # held-out split: different seed than default training run
        x, y = synthetic_dataset(64, seed=seed + 1000)

    model = build_model(weights)
    with torch.no_grad():
        pred = model(x)
    metrics = _confusion(pred, y)

    if pte:
        from .run_pte import run_pte

        # whole-set parity: run each input through the runtime, compare to eager
        max_err = 0.0
        for i in range(x.shape[0]):
            out = run_pte(pte, x[i:i + 1])
            max_err = max(max_err, (out - pred[i:i + 1]).abs().max().item())
        metrics["pte_max_abs_err"] = max_err

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate StressNet")
    ap.add_argument("--weights", "-w", required=True, help="checkpoint .pt")
    ap.add_argument("--data-dir", default=None, help="real eval data root")
    ap.add_argument("--pte", default=None, help="also fleet-parity-check this .pte")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = evaluate(args.weights, data_dir=args.data_dir, pte=args.pte, seed=args.seed)
    print(f"accuracy : {m['accuracy']:.3f}")
    print(f"precision: {m['precision']:.3f}   recall: {m['recall']:.3f}   f1: {m['f1']:.3f}")
    print(f"confusion: tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']}")
    if "pte_max_abs_err" in m:
        print(f"pte parity (max abs err over set): {m['pte_max_abs_err']:.2e}")


if __name__ == "__main__":
    main()
