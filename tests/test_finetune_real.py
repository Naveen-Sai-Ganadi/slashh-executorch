"""Tests for real-data fine-tuning (model/finetune_real.py + model/realdata.py).

Pure-logic tests need no audio. The end-to-end path synthesizes a handful of
PCM wavs in a tmp dir (stdlib ``wave``), so it exercises the real loader +
speaker-independent split + train loop without any downloaded corpus.

Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import struct
import wave as wavmod

import pytest
import torch


# --- speaker-independent split ---------------------------------------------

def test_speaker_split_disjoint():
    from model.finetune_real import speaker_split

    speakers = [f"ravdess:{i//4:02d}" for i in range(40)]  # 10 speakers, 4 each
    tr, val = speaker_split(speakers, val_fraction=0.25, seed=0)
    tr_spk = {speakers[i] for i in tr}
    val_spk = {speakers[i] for i in val}
    assert tr_spk and val_spk
    assert tr_spk.isdisjoint(val_spk)            # no speaker in both
    assert len(tr) + len(val) == len(speakers)   # partition, no loss


def test_speaker_split_keeps_each_corpus_in_both():
    from model.finetune_real import speaker_split

    speakers = [f"ravdess:{i:02d}" for i in range(8)] + \
               [f"emodb:{i:02d}" for i in range(8)]
    tr, val = speaker_split(speakers, val_fraction=0.25, seed=1)
    corp = lambda idx: {speakers[i].split(":")[0] for i in idx}
    assert corp(tr) == {"ravdess", "emodb"}
    assert corp(val) == {"ravdess", "emodb"}


# --- metrics ----------------------------------------------------------------

def test_evaluate_perfect():
    from model.finetune_real import evaluate

    class Const(torch.nn.Module):
        def __init__(self, y):
            super().__init__()
            self.y = y
        def forward(self, x):
            return self.y

    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    m = evaluate(Const(y), torch.zeros(4, 1, 64, 301), y,
                 ["a:1", "a:1", "b:2", "b:2"], threshold=0.5)
    assert m.accuracy == pytest.approx(1.0)
    assert m.tp == 2 and m.tn == 2 and m.fp == 0 and m.fn == 0


def test_best_accuracy_threshold_separates():
    from model.finetune_real import best_accuracy_threshold

    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    t = best_accuracy_threshold(scores, y)
    assert 0.2 < t < 0.8                          # any cut between the clusters


def test_class_weights_balance():
    from model.finetune_real import _class_weights

    y = torch.tensor([[1.0], [1.0], [1.0], [0.0]])  # 75% positive
    w = _class_weights(y)
    # total weight on each class should be equal after inverse-freq weighting.
    assert float(w[y.flatten() == 1].sum()) == pytest.approx(
        float(w[y.flatten() == 0].sum()), abs=1e-5
    )


# --- end-to-end on synthesized wavs ----------------------------------------

def _write_wav(path, freq, sr=16000, secs=1.0):
    n = int(sr * secs)
    frames = bytearray()
    for i in range(n):
        # crude tone; exact content irrelevant — we only need a valid PCM wav.
        v = int(12000 * ((i * freq // sr) % 2 * 2 - 1))
        frames += struct.pack("<h", v)
    with wavmod.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def _make_ravdess(root, n_actors=6):
    """Emotion 05 (angry->stressed) and 02 (calm) for several actors."""
    for actor in range(1, n_actors + 1):
        for emo in ("05", "02"):
            for rep in ("01", "02"):
                name = f"03-01-{emo}-01-01-{rep}-{actor:02d}.wav"
                _write_wav(root / name, freq=300 if emo == "05" else 120)


def test_build_real_dataset_and_finetune(tmp_path):
    from model.realdata import build_real_dataset
    from model.finetune_real import finetune

    root = tmp_path / "ravdess"
    root.mkdir()
    _make_ravdess(root)
    ds = build_real_dataset({"ravdess": root})
    assert ds.x.shape[1:] == (1, 64, 301)
    assert len(ds) == ds.x.shape[0] == ds.y.shape[0]
    assert set(ds.speakers) and torch.isfinite(ds.x).all()

    # fine-tune from random init (no checkpoint needed) for a couple epochs.
    model, res = finetune(ds, init_weights=None, epochs=2, seed=0, batch_size=8)
    assert 0.0 <= res.val.accuracy <= 1.0
    assert res.val.n > 0                          # held-out speakers exist
    assert 0.0 < res.tuned_threshold < 1.0


def test_build_finetune_writes_records(tmp_path):
    from model.realdata import build_real_dataset  # noqa: F401 (import smoke)
    from model.finetune_real import build_finetune

    root = tmp_path / "ravdess"
    root.mkdir()
    _make_ravdess(root)
    out = build_finetune(
        ravdess=root, emodb=None, init_weights=None, epochs=2, seed=0,
        out_dir=tmp_path / "bench",
    )
    assert (tmp_path / "bench" / "finetune_real.json").is_file()
    assert (tmp_path / "bench" / "finetune_real.md").is_file()
    assert out.val.n > 0
