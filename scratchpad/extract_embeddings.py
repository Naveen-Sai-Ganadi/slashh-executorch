"""Extract & cache frozen-backbone embeddings over the unified corpora.

The SOTA/zoo backbones consume RAW 16kHz waveforms (not our 64-mel log-mel),
so we re-walk the corpora with the *same* parser functions used to build
unified.pt (guaranteeing identical sample order), load each raw 3s/16kHz
waveform via model.data._load_wave, run the backbone, and cache the pooled
embedding aligned to the unified dataset.

Output per backbone: datasets/unified/emb_<name>.pt =
  {"emb": [N, D] float32, "speaker": [...], "corpus": [...],
   "y_narrow": [N,1], "y_broad": [N,1], "split": [...]}

Run one backbone at a time (each loads a ~300M model):
  python scratchpad/extract_embeddings.py whisper
  python scratchpad/extract_embeddings.py audeering
  python scratchpad/extract_embeddings.py wavlm
  python scratchpad/extract_embeddings.py hubert
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch

from model.data import _load_wave

SR = 16000
ROOT = Path("datasets")
OUTDIR = ROOT / "unified"

# --- canonical labels + parsers (kept identical to the unified builder so the
#     sample order matches unified.pt exactly) ------------------------------
NARROW_STRESSED = {"angry", "fearful"}
NARROW_CALM = {"neutral", "calm"}
BROAD_STRESSED = {"angry", "fearful", "happy", "surprised", "disgust"}

RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
CREMA_EMO = {"NEU": "neutral", "HAP": "happy", "SAD": "sad",
             "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
TESS_EMO = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
            "fear": "fearful", "disgust": "disgust", "ps": "surprised"}
SAVEE_EMO = {"n": "neutral", "h": "happy", "sa": "sad", "a": "angry",
             "f": "fearful", "d": "disgust", "su": "surprised"}

VAL_SPEAKERS = set()
VAL_SPEAKERS |= {f"RAV_{a:02d}" for a in (21, 22, 23, 24)}
VAL_SPEAKERS |= {f"CRE_{1000 + i}" for i in range(74, 92)}
VAL_SPEAKERS |= {"SAV_KL"}


def ravdess_items():
    for w in sorted((ROOT / "ravdess").rglob("*.wav")):
        p = w.stem.split("-")
        if len(p) != 7:
            continue
        emo = RAVDESS_EMO.get(p[2])
        if emo is None:
            continue
        actor = int(p[6])
        gender = "male" if actor % 2 == 1 else "female"
        yield w, emo, f"RAV_{actor:02d}", "RAVDESS", gender


def crema_items():
    for w in sorted((ROOT / "crema-d").rglob("*.wav")):
        p = w.stem.split("_")
        if len(p) < 3:
            continue
        emo = CREMA_EMO.get(p[2])
        if emo is None:
            continue
        yield w, emo, f"CRE_{p[0]}", "CREMA-D", "unk"


def tess_items():
    for w in sorted((ROOT / "tess").rglob("*.wav")):
        folder = w.parent.name
        spk = folder.split("_")[0]
        key = folder.split("_", 1)[1].lower()
        if "surpris" in key:
            emo = "surprised"
        elif "fear" in key:
            emo = "fearful"
        else:
            emo = TESS_EMO.get(key, key)
        yield w, emo, f"TESS_{spk}", "TESS", "female"


def savee_items():
    for w in sorted((ROOT / "savee").rglob("*.wav")):
        spk, _, rest = w.stem.partition("_")
        code = "".join(c for c in rest if c.isalpha())
        emo = SAVEE_EMO.get(code)
        if emo is None:
            continue
        yield w, emo, f"SAV_{spk}", "SAVEE", "male"


def all_items():
    for src in (ravdess_items, crema_items, tess_items, savee_items):
        yield from src()


def labels(emo):
    if emo in NARROW_STRESSED:
        yn = 1.0
    elif emo in NARROW_CALM:
        yn = 0.0
    else:
        yn = -1.0
    yb = 1.0 if emo in BROAD_STRESSED else 0.0
    return yn, yb


# ---- backbone embedders: waveform [T] float32 -> embedding [D] -------------
def make_whisper():
    # HF Whisper encoder gives a clean (1, 1500, 384) hidden state we can pool
    # for head training. Shares weights with the qai_hub whisper-tiny we export
    # on-device; the qai_hub encoder forward emits decoder KV-cache, not a
    # hidden state, so it's unsuitable for embedding extraction.
    from transformers import WhisperModel, WhisperFeatureExtractor
    model = WhisperModel.from_pretrained("openai/whisper-tiny")
    enc = model.encoder
    enc.eval()
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def embed(wave):
        feats = fe(wave.numpy(), sampling_rate=SR, return_tensors="pt").input_features
        with torch.no_grad():
            out = enc(feats)
        h = out.last_hidden_state            # (1, 1500, 384)
        return h.mean(dim=1).squeeze(0)      # 384-d
    return embed


def _hf_embedder(name, kind):
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(name)
    if kind == "cls":
        from transformers import AutoModelForAudioClassification
        model = AutoModelForAudioClassification.from_pretrained(name)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name)
    model.eval()

    def embed(wave):
        inp = fe(wave.numpy(), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=(kind == "cls"))
        if kind == "cls":
            # last hidden state mean-pool = a stable feature; logits also useful
            h = out.hidden_states[-1]
            return h.mean(dim=1).squeeze(0)
        return out.last_hidden_state.mean(dim=1).squeeze(0)
    return embed


BACKBONES = {
    "whisper": make_whisper,
    "audeering": lambda: _hf_embedder("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "cls"),
    "wavlm": lambda: _hf_embedder("microsoft/wavlm-large", "feat"),
    "hubert": lambda: _hf_embedder("superb/hubert-large-superb-er", "cls"),
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "whisper"
    embed = BACKBONES[name]()
    t0 = time.time()
    embs, spks, corps, yns, ybs, splits = [], [], [], [], [], []
    n = 0
    for w, emo, spk, corpus, gender in all_items():
        wave = _load_wave(w)                  # [1,1,?]->? actually returns [T]
        wave = wave.squeeze()
        e = embed(wave).float().cpu()
        embs.append(e)
        yn, yb = labels(emo)
        spks.append(spk); corps.append(corpus)
        yns.append(yn); ybs.append(yb)
        splits.append("val" if spk in VAL_SPEAKERS else "train")
        n += 1
        if n % 500 == 0:
            print(f"  {name}: {n} ({time.time()-t0:.0f}s)", flush=True)
    emb = torch.stack(embs)                   # [N, D]
    out = OUTDIR / f"emb_{name}.pt"
    torch.save({
        "emb": emb,
        "speaker": spks, "corpus": corps,
        "y_narrow": torch.tensor(yns).unsqueeze(1),
        "y_broad": torch.tensor(ybs).unsqueeze(1),
        "split": splits,
        "backbone": name,
    }, out)
    print(f"saved {out}  emb={list(emb.shape)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
