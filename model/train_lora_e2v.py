"""LoRA finetuning for the emotion2vec teacher (4th of the SOTA roster).

emotion2vec ships via funasr, not transformers, and is a data2vec-style encoder
whose attention/MLP linears are named differently from the wav2vec2 family
(``attn.qkv`` / ``attn.proj`` / ``mlp.fc1`` / ``mlp.fc2`` under
``modality_encoders.AUDIO.context_encoder.blocks.{i}``). So it gets its own
trainer, run in the isolated ``.venv_e2v`` (funasr would churn the live env's
pins), but on the SAME unified speaker-independent split, class-weighted
objective and balanced-accuracy selection as every other model in the roster.

The differentiable forward is ``model.extract_features(wave)["x"]`` -> [B,T,768];
peft injects LoRA into the matched linears in-place, so we call extract_features
on the wrapped encoder and only the adapters + stress head update.

    .venv_e2v/bin/python3 -m model.train_lora_e2v --mapping broad --epochs 5

Writes model/checkpoints/lora_emotion2vec_<map>/adapter_head.pt and
docs/benchmarks/lora_emotion2vec_<map>.json (folded into the leaderboard).
Host-only; never touches the live AI Hub token.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

UNIFIED = Path("datasets/unified/unified.pt")
WAVE_CACHE = Path("datasets/unified/waves_16k.pt")
CKPT = Path("model/checkpoints")
BENCH = Path("docs/benchmarks")

E2V_MODEL = "iic/emotion2vec_plus_base"
# attention qkv/proj + MLP fc1/fc2 across every transformer block
LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
FEAT_DIM = 768  # emotion2vec_plus_base hidden size


def _norm(w: torch.Tensor) -> torch.Tensor:
    """Per-utterance zero-mean unit-var — matches emotion2vec's input layer_norm."""
    return (w - w.mean(dim=-1, keepdim=True)) / (w.std(dim=-1, keepdim=True) + 1e-7)


def _split_idx(mapping: str):
    d = torch.load(UNIFIED, weights_only=False)
    y = d["y_narrow" if mapping == "narrow" else "y_broad"].float().squeeze(1)
    is_val = torch.tensor([s == "val" for s in d["split"]])
    keep = y >= 0
    tr = torch.nonzero((~is_val) & keep).squeeze(1)
    va = torch.nonzero(is_val & keep).squeeze(1)
    return tr, va, y


def _bal_acc(logits: torch.Tensor, y: torch.Tensor):
    p = (torch.sigmoid(logits.flatten()) > 0.5).float()
    yy = y.flatten()
    per = []
    for c in (0.0, 1.0):
        m = yy == c
        if m.any():
            per.append((p[m] == yy[m]).float().mean().item())
    acc = (p == yy).float().mean().item()
    return acc, (sum(per) / len(per) if per else 0.0)


class LoRAE2V(nn.Module):
    """LoRA-adapted emotion2vec encoder + small stress head -> logit."""

    def __init__(self, r: int = 8, alpha: int = 16):
        super().__init__()
        from funasr import AutoModel
        from peft import LoraConfig, get_peft_model

        enc = AutoModel(model=E2V_MODEL, disable_update=True).model
        lcfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                          lora_dropout=0.05, bias="none")
        # get_peft_model injects adapters in-place and freezes everything else;
        # we keep the wrapper (for the adapter state_dict) but call the encoder's
        # own extract_features in forward.
        self.peft = get_peft_model(enc, lcfg)
        self.enc = enc
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def forward(self, wave):  # [B,48000] (already per-utterance normalized)
        feat = self.enc.extract_features(wave, padding_mask=None)["x"]  # [B,T,768]
        return self.head(feat.mean(dim=1))                             # [B,1] logit

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.peft.state_dict().items() if "lora" in k}


@torch.no_grad()
def _evaluate(model, waves, idx, y, device, bs=16):
    model.eval()
    outs = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = _norm(waves[b]).to(device)
        outs.append(model(x).float().cpu())
    return _bal_acc(torch.cat(outs), y[idx])


def train(mapping: str = "broad", epochs: int = 5, bs: int = 8,
          lr_lora: float = 1e-4, lr_head: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[lora:emotion2vec/{mapping}] device={device}", flush=True)

    waves = torch.load(WAVE_CACHE, weights_only=False)["waves"]
    tr, va, y = _split_idx(mapping)
    pos = float(y[tr].sum()); neg = float(len(tr) - y[tr].sum())
    pos_w = torch.tensor([neg / max(pos, 1.0)], device=device)
    print(f"  train={len(tr)} val={len(va)} pos_w={pos_w.item():.3f}", flush=True)

    model = LoRAE2V().to(device)
    n_train = sum(p.numel() for p in model.trainable_params())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {n_train/1e6:.2f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.2f}%)", flush=True)

    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora" in n], "lr": lr_lora},
        {"params": model.head.parameters(), "lr": lr_head},
    ], weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    gen = torch.Generator().manual_seed(seed)
    best_bal, best, best_state = 0.0, {}, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr), generator=gen)]
        run = 0.0
        for bi, i in enumerate(range(0, len(order), bs)):
            b = order[i:i + bs]
            x = _norm(waves[b]).to(device)
            yb = y[b].unsqueeze(1).to(device)
            opt.zero_grad()
            loss = bce(model(x), yb)
            loss.backward()
            opt.step()
            run += loss.item()
            if (bi + 1) % 100 == 0:
                print(f"    ep{ep} step{bi+1}/{len(order)//bs} "
                      f"loss={run/(bi+1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        acc, bal = _evaluate(model, waves, va, y, device)
        print(f"  [ep{ep}] val_acc={acc:.4f} val_bal_acc={bal:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if bal >= best_bal:
            best_bal = bal
            best = {"val_acc": acc, "val_bal_acc": bal, "epoch": ep}
            best_state = {"head": {k: v.detach().cpu().clone()
                                   for k, v in model.head.state_dict().items()},
                          "lora": model.adapter_state()}

    out = {
        "backbone": "emotion2vec", "mapping": mapping, "method": "lora",
        "lora_r": 8, "lora_alpha": 16, "targets": LORA_TARGETS,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "pos_weight": pos_w.item(), "epochs": epochs,
        "trainable_params": int(n_train), "total_params": int(n_total),
        **best,
    }
    return model, best_state, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", choices=["narrow", "broad"], default="broad")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    model, state, meta = train(args.mapping, args.epochs, args.bs)

    tag = f"emotion2vec_{args.mapping}"
    d = CKPT / f"lora_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state, d / "adapter_head.pt")
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / f"lora_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[lora:{tag}] val_bal_acc={meta.get('val_bal_acc',0):.3f} "
          f"trainable={meta['trainable_params']/1e6:.2f}M -> {d}")


if __name__ == "__main__":
    main()
