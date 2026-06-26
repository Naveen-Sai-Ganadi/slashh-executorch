"""Tests for the cross-initialization robustness envelope (model/init_envelope.py).

The README stakes a headline claim: the shipped net is small enough that its
noise floor is *init-sensitive*, so "10 dB is the envelope we stand on." That is
a claim about variance across random initializations — which the eval-seed
averaging in model/robustness.py does NOT measure (it varies the eval draw with
the model fixed). This harness trains several independent inits, takes each
one's reliable floor, and derives the **conservative envelope**: the SNR every
init is still reliable down to. Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.init_envelope import InitEnvelopeResult, init_envelope
from model.model import StressNet
from model.train import train


def _trained(seed: int) -> StressNet:
    net = StressNet(channels=(4, 8, 16))
    train(data_dir=None, epochs=10, batch_size=16, lr=1e-3, n_per_class=48,
          seed=seed, model=net)
    return net.eval()


def test_collects_one_floor_per_init_and_clean_acc() -> None:
    models = [(s, _trained(s)) for s in (0, 1)]
    out = init_envelope(
        models, snr_levels=[None, 20.0, 10.0, 0.0], n_per_class=24,
        eval_seed=7, threshold=0.8,
    )
    assert isinstance(out, InitEnvelopeResult)
    assert [f.seed for f in out.floors] == [0, 1]
    for f in out.floors:
        assert 0.0 <= f.clean_acc <= 1.0
        # a reliable floor is either None (clean-only) or one of the noisy SNRs
        assert f.reliable_floor_db in {None, 20.0, 10.0, 0.0}


def test_envelope_is_the_worst_reliable_floor_across_inits() -> None:
    models = [(s, _trained(s)) for s in (0, 1, 2)]
    out = init_envelope(
        models, snr_levels=[None, 20.0, 10.0, 0.0, -10.0], n_per_class=24,
        eval_seed=7, threshold=0.8,
    )
    floors = [f.reliable_floor_db for f in out.floors]
    if any(v is None for v in floors):
        # if any init is reliable only on clean audio, there is no common noisy
        # floor and the envelope is undefined (None)
        assert out.envelope_db is None
    else:
        # the envelope is the *highest* (least-noisy) per-init floor — the SNR
        # every init still clears
        assert out.envelope_db == max(floors)


def test_deterministic_across_identical_calls() -> None:
    models = [(s, _trained(s)) for s in (0, 1)]
    a = init_envelope(models, snr_levels=[None, 10.0, 0.0], n_per_class=20, eval_seed=3)
    b = init_envelope(models, snr_levels=[None, 10.0, 0.0], n_per_class=20, eval_seed=3)
    assert a.to_dict() == b.to_dict()


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    models = [(s, _trained(s)) for s in (0, 1)]
    init_envelope(
        models, snr_levels=[None, 10.0, 0.0], n_per_class=20, eval_seed=3,
        out_dir=tmp_path,
    )
    assert (tmp_path / "init_envelope.json").is_file()
    assert (tmp_path / "init_envelope.md").is_file()
    payload = json.loads((tmp_path / "init_envelope.json").read_text())
    assert "envelope_db" in payload and "floors" in payload
    assert len(payload["floors"]) == 2
    md = (tmp_path / "init_envelope.md").read_text()
    assert "envelope" in md.lower()
