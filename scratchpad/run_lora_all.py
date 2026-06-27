"""Drive LoRA finetuning for every HF teacher backbone, sequentially.

MPS runs one ~317M model at a time, so we train the three wav2vec2-family
teachers back-to-back rather than in parallel. Each backbone loads fresh in its
own train() call (no cross-model memory accumulation). emotion2vec is handled by
a separate path (different loader) and is intentionally not in this list.

  PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/run_lora_all.py broad
"""
import sys
import json
import time
import traceback
from pathlib import Path

import torch

from model.train_lora import train, CKPT, BENCH

BACKBONES = ["wavlm", "hubert", "audeering"]


def main():
    mapping = sys.argv[1] if len(sys.argv) > 1 else "broad"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    BENCH.mkdir(parents=True, exist_ok=True)
    for bb in BACKBONES:
        t0 = time.time()
        print(f"\n========== LoRA {bb}/{mapping} ({epochs} ep) ==========", flush=True)
        try:
            model, state, meta = train(bb, mapping, epochs=epochs)
            tag = f"{bb}_{mapping}"
            d = CKPT / f"lora_{tag}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save(state, d / "adapter_head.pt")
            (BENCH / f"lora_{tag}.json").write_text(json.dumps(meta, indent=2))
            print(f"DONE {tag} val_bal_acc={meta.get('val_bal_acc',0):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model, state
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            print(f"FAILED {bb}/{mapping}:\n{traceback.format_exc()}", flush=True)
    print("\nALL_LORA_DONE", flush=True)


if __name__ == "__main__":
    main()
