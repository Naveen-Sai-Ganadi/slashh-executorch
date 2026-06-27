"""Build ONE clean unified dataset from the 4 emotion-speech corpora.

Reuses the exact, already-tested feature front-end from model/data.py
(_load_wave -> soundfile load + mono + resample-to-16kHz + center-crop/pad to
WINDOW_SAMPLES -> extract() -> [1,1,64,301]). No info loss: every sample keeps
its raw canonical emotion, BOTH arousal mappings (narrow + broad), speaker id,
corpus, gender, and a speaker-independent split assignment.

Output artifact: datasets/unified/unified.pt

Run with both repo root and this dir on PYTHONPATH:
  PYTHONPATH="$PWD:$PWD/scratchpad" .venv/bin/python3 scratchpad/build_unified.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from model.data import _load_wave
from model.features import extract

ROOT = Path("datasets")
OUT = ROOT / "unified" / "unified.pt"

NARROW_STRESSED = {"angry", "fearful"}
NARROW_CALM = {"neutral", "calm"}
BROAD_STRESSED = {"angry", "fearful", "happy", "surprised", "disgust"}
BROAD_CALM = {"neutral", "calm", "sad"}

RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
CREMA_EMO = {"NEU": "neutral", "HAP": "happy", "SAD": "sad",
             "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
TESS_EMO = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
            "fear": "fearful", "disgust": "disgust", "ps": "surprised"}
SAVEE_EMO = {"n": "neutral", "h": "happy", "sa": "sad", "a": "angry",
             "f": "fearful", "d": "disgust", "su": "surprised"}


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


VAL_SPEAKERS = set()
VAL_SPEAKERS |= {f"RAV_{a:02d}" for a in (21, 22, 23, 24)}
VAL_SPEAKERS |= {f"CRE_{1000 + i}" for i in range(74, 92)}
VAL_SPEAKERS |= {"SAV_KL"}


def main():
    t0 = time.time()
    feats, emos, spks, corps, genders = [], [], [], [], []
    sources = [ravdess_items, crema_items, tess_items, savee_items]
    n = 0
    for src in sources:
        cn = 0
        for w, emo, spk, corpus, gender in src():
            feats.append(extract(_load_wave(w)))
            emos.append(emo); spks.append(spk); corps.append(corpus); genders.append(gender)
            n += 1; cn += 1
            if n % 1000 == 0:
                print(f"  {n} files... ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{src.__name__}] {cn} usable", flush=True)

    x = torch.cat(feats, dim=0)

    def narrow(e):
        if e in NARROW_STRESSED:
            return 1.0
        if e in NARROW_CALM:
            return 0.0
        return -1.0

    def broad(e):
        return 1.0 if e in BROAD_STRESSED else 0.0

    y_narrow = torch.tensor([narrow(e) for e in emos], dtype=torch.float32).unsqueeze(1)
    y_broad = torch.tensor([broad(e) for e in emos], dtype=torch.float32).unsqueeze(1)
    split = ["val" if s in VAL_SPEAKERS else "train" for s in spks]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": x, "emotion": emos, "y_narrow": y_narrow, "y_broad": y_broad,
        "speaker": spks, "corpus": corps, "gender": genders, "split": split,
        "meta": {
            "n": n, "feature_shape": list(x.shape), "sample_rate": 16000,
            "window_samples": 48000,
            "narrow_stressed": sorted(NARROW_STRESSED), "narrow_calm": sorted(NARROW_CALM),
            "broad_stressed": sorted(BROAD_STRESSED), "broad_calm": sorted(BROAD_CALM),
            "val_speakers": sorted(VAL_SPEAKERS),
        },
    }, OUT)
    print(f"\nsaved {OUT}  N={n}  shape={list(x.shape)}  "
          f"({time.time()-t0:.0f}s, {OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
