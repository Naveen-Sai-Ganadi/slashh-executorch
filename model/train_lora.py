"""LoRA finetuning for the SOTA teacher backbones (Track B).

The four teacher encoders — audeering wav2vec2, microsoft WavLM, HuBERT-SER,
and emotion2vec — are too large for on-device. Rather than the frozen-backbone
+ head linear probe, we attach low-rank adapters (LoRA) to every attention
projection and train them *together* with a small stress head end-to-end on raw
16 kHz waveforms. This is true finetuning (the backbone's effective weights move
via the adapters) at a fraction of the trainable-parameter cost, so it fits on
the M4 Pro's MPS.

Each backbone shares the wav2vec2 attention lineage (q/k/v/out_proj), so one
config covers all three HF models; emotion2vec is handled separately.

Trained on the unified speaker-independent split with the SAME class-weighted
objective and balanced-accuracy selection used for StressNet, so every model in
the roster is judged on identical held-out speakers.

  PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 -m model.train_lora \
      --backbone wavlm --mapping broad --epochs 4

Writes model/checkpoints/lora_<bb>_<map>/ (adapter + head + norm) and
docs/benchmarks/lora_<bb>_<map>.json (folded into the leaderboard).
Host-only; never touches the live AI Hub token.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

# NOTE: the real-audio single-file loader (`model.data._load_wave`) is imported
# lazily inside `load_waves()` — it is only needed on a wave-cache MISS. data.py
# was refactored to the synthetic-only API, so a top-level import would crash
# every consumer (export/parity/quant) even when the 16 kHz wave cache exists.

UNIFIED = Path("datasets/unified/unified.pt")
WAVE_CACHE = Path("datasets/unified/waves_16k.pt")
CKPT = Path("model/checkpoints")
BENCH = Path("docs/benchmarks")

# HF base encoders. AutoModel strips any task head, leaving the bare encoder
# whose last_hidden_state we mean-pool. All three are wav2vec2-family.
HF_BASE = {
    "audeering": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "wavlm": "microsoft/wavlm-large",
    "hubert": "superb/hubert-large-superb-er",
}
# every wav2vec2/wavlm/hubert attention block names its projections this way
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


# --- waveform cache (shared across backbones, built once) -------------------
def _items():
    """Walk corpora in the EXACT order build_unified.py used, so waves align
    index-for-index with unified.pt's split / y_narrow / y_broad."""
    from scratchpad.extract_embeddings import all_items
    return list(all_items())


def load_waves() -> torch.Tensor:
    """[N, WIN] float32, fixed 3 s / 16 kHz windows. Cached to disk."""
    if WAVE_CACHE.exists():
        return torch.load(WAVE_CACHE, weights_only=False)["waves"]
    t0 = time.time()
    from model.data import _load_wave  # lazy: only needed to rebuild the cache
    items = _items()
    waves = torch.empty(len(items), 48000, dtype=torch.float32)
    for i, (w, *_rest) in enumerate(items):
        waves[i] = _load_wave(w).squeeze()
        if (i + 1) % 1000 == 0:
            print(f"  wave cache {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    WAVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"waves": waves, "n": len(items)}, WAVE_CACHE)
    print(f"saved {WAVE_CACHE}  {list(waves.shape)} ({time.time() - t0:.0f}s)", flush=True)
    return waves


def _norm(w: torch.Tensor) -> torch.Tensor:
    """Per-utterance zero-mean unit-var — matches Wav2Vec2FeatureExtractor
    do_normalize=True, applied here so we skip per-batch FE overhead."""
    return (w - w.mean(dim=-1, keepdim=True)) / (w.std(dim=-1, keepdim=True) + 1e-7)


# --- model ------------------------------------------------------------------
class LoRAStress(nn.Module):
    """LoRA-adapted teacher encoder + small stress head -> logit."""

    def __init__(self, base_name: str, r: int = 8, alpha: int = 16):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        enc = AutoModel.from_pretrained(base_name)
        enc.freeze_feature_encoder()  # conv frontend stays fixed (standard)
        lcfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                          lora_dropout=0.05, bias="none")
        self.encoder = get_peft_model(enc, lcfg)
        hidden = enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def forward(self, input_values):
        h = self.encoder(input_values).last_hidden_state   # [B,T,H]
        return self.head(h.mean(dim=1))                    # [B,1] logit

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


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


@torch.no_grad()
def _evaluate(model, waves, idx, y, device, bs=16):
    model.eval()
    outs = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = _norm(waves[b]).to(device)
        outs.append(model(x).float().cpu())
    logits = torch.cat(outs)
    return _bal_acc(logits, y[idx])


def train(backbone: str, mapping: str = "broad", epochs: int = 4,
          bs: int = 8, lr_lora: float = 1e-4, lr_head: float = 1e-3,
          seed: int = 0, eval_every: int = 1):
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[lora:{backbone}/{mapping}] device={device}", flush=True)

    waves = load_waves()
    tr, va, y = _split_idx(mapping)
    pos = float(y[tr].sum()); neg = float(len(tr) - y[tr].sum())
    pos_w = torch.tensor([neg / max(pos, 1.0)], device=device)
    print(f"  train={len(tr)} val={len(va)} pos_w={pos_w.item():.3f}", flush=True)

    model = LoRAStress(HF_BASE[backbone]).to(device)
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
    best_bal, best = 0.0, {}
    best_state = None
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
        if ep % eval_every == 0 or ep == epochs:
            acc, bal = _evaluate(model, waves, va, y, device)
            print(f"  [ep{ep}] val_acc={acc:.4f} val_bal_acc={bal:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if bal >= best_bal:
                best_bal = bal
                best = {"val_acc": acc, "val_bal_acc": bal, "epoch": ep}
                best_state = {
                    "head": {k: v.detach().cpu().clone()
                             for k, v in model.head.state_dict().items()},
                    "lora": {k: v.detach().cpu().clone()
                             for k, v in model.encoder.state_dict().items()
                             if "lora" in k},
                }

    out = {
        "backbone": backbone, "mapping": mapping, "method": "lora",
        "lora_r": 8, "lora_alpha": 16, "targets": LORA_TARGETS,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "pos_weight": pos_w.item(), "epochs": epochs,
        "trainable_params": int(n_train), "total_params": int(n_total),
        **best,
    }
    return model, best_state, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(HF_BASE))
    ap.add_argument("--mapping", choices=["narrow", "broad"], default="broad")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    model, state, meta = train(args.backbone, args.mapping, args.epochs, args.bs)

    tag = f"{args.backbone}_{args.mapping}"
    d = CKPT / f"lora_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state, d / "adapter_head.pt")
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / f"lora_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[lora:{tag}] val_acc={meta.get('val_acc',0):.3f} "
          f"val_bal_acc={meta.get('val_bal_acc',0):.3f} "
          f"trainable={meta['trainable_params']/1e6:.2f}M -> {d}")


if __name__ == "__main__":
    main()
