"""Verify the SOTA SER backbones load and produce embeddings/arousal on a real
unified-dataset waveform-equivalent. We feed a 3s, 16kHz dummy waveform (the
models consume raw audio, not our log-mel) and report output shapes.

These are frozen-backbone teachers/reference models — we only need them to (a)
load and (b) emit a stable feature vector (or arousal scalar) we can cache.
"""
from __future__ import annotations
import traceback
import torch

SR = 16000
WAV = torch.zeros(1, 3 * SR)  # 3s silence — shape probe only


def probe_audeering():
    # outputs arousal/dominance/valence DIRECTLY (regression head)
    from transformers import Wav2Vec2Processor, AutoModelForAudioClassification
    name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    proc = Wav2Vec2Processor.from_pretrained(name)
    model = AutoModelForAudioClassification.from_pretrained(name)
    model.eval()
    inp = proc(WAV.squeeze(0).numpy(), sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    logits = getattr(out, "logits", None)
    print(f"  audeering OK  logits={tuple(logits.shape) if logits is not None else out}")


def probe_wavlm():
    from transformers import AutoFeatureExtractor, WavLMModel
    name = "microsoft/wavlm-large"
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = WavLMModel.from_pretrained(name)
    model.eval()
    inp = fe(WAV.squeeze(0).numpy(), sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    h = out.last_hidden_state
    print(f"  wavlm-large OK  hidden={tuple(h.shape)}  pooled={tuple(h.mean(1).shape)}")


def probe_hubert_ser():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    name = "superb/hubert-large-superb-er"  # emotion-recognition SER head
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModelForAudioClassification.from_pretrained(name)
    model.eval()
    inp = fe(WAV.squeeze(0).numpy(), sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    print(f"  hubert-SER OK  logits={tuple(out.logits.shape)}  labels={model.config.id2label}")


for name, fn in [("audeering", probe_audeering),
                 ("wavlm", probe_wavlm),
                 ("hubert_ser", probe_hubert_ser)]:
    print(f"[{name}] loading...", flush=True)
    try:
        fn()
    except Exception:
        print(f"[{name}] FAILED")
        traceback.print_exc()
print("done")
