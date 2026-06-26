"""fp32 -> INT8 confidence-calibration drift (TDD, red first).

The shipped ``DEFAULT_TEMPERATURE = 0.41`` and the whole score-calibration story
(``model.calibration``) were measured on the *fp32 eager* model. But the device
serves the **INT8 ``.pte``** (``quantize_production``). PT2E quantization perturbs
the activations, so the sigmoid scores the user actually sees on-device can have a
different reliability profile than the one we calibrated. If the score
distribution shifts, the calibrated-confidence read-out (and the fp32-fitted
temperature) may be miscalibrated on the deployed model.

This harness collects scores from the *same* eval through both the fp32 eager
model and its INT8 ``.pte`` runtime, computes calibration (ECE + fitted
temperature) for each, and asks two questions: (1) does INT8 drift the ECE, and
(2) does the fp32-fitted shipped temperature still reduce INT8 ECE, or must it be
re-fit on the deployed model?

Host-only; no device, no AI Hub token. Pure reduction is unit-tested without
training; the heavy ``build_*`` path trains one tiny net and quantizes it.
"""

from __future__ import annotations

import json

import pytest
import torch


# --- pure reduction logic (no training) -----------------------------------

def _scores(n, hedge, *, seed, flip=0):
    """Synthetic (score, label) pairs with a tunable under-confidence ``hedge``.

    ``hedge`` in (0,1] pulls correct scores toward 0.5 (smaller = more hedged =
    worse-calibrated). ``flip`` mislabels the first ``flip`` samples so accuracy
    and the reliability gap are non-degenerate.
    """
    g = torch.Generator().manual_seed(seed)
    labels = (torch.arange(n) % 2).float()
    base = labels.clone()
    if flip:
        base[:flip] = 1.0 - base[:flip]
    scores = 0.5 + (base - 0.5) * hedge
    scores = (scores + 0.02 * torch.randn(n, generator=g)).clamp(0.01, 0.99)
    return scores, labels


def test_reports_per_model_ece_and_drift():
    from model.int8_calibration_drift import int8_calibration_drift
    from model.calibration import compute_calibration

    fp32_s, y = _scores(200, hedge=0.6, seed=0, flip=20)
    int8_s, _ = _scores(200, hedge=0.5, seed=1, flip=20)
    out = int8_calibration_drift(fp32_s, y, int8_s, y)

    fp32_ece = compute_calibration(fp32_s, y, fit_temperature=False).ece
    int8_ece = compute_calibration(int8_s, y, fit_temperature=False).ece
    assert out.fp32.ece == pytest.approx(fp32_ece, abs=1e-9)
    assert out.int8.ece == pytest.approx(int8_ece, abs=1e-9)
    assert out.ece_drift == pytest.approx(int8_ece - fp32_ece, abs=1e-9)


def test_shipped_temperature_defaults_to_constant():
    from model.int8_calibration_drift import int8_calibration_drift
    from model.confidence import DEFAULT_TEMPERATURE

    fp32_s, y = _scores(120, hedge=0.6, seed=0, flip=12)
    int8_s, _ = _scores(120, hedge=0.55, seed=1, flip=12)
    out = int8_calibration_drift(fp32_s, y, int8_s, y)
    assert out.shipped_temperature == pytest.approx(DEFAULT_TEMPERATURE)


def test_shipped_temperature_transfer_ece_is_reported():
    from model.int8_calibration_drift import int8_calibration_drift
    from model.confidence import temperature_scale
    from model.calibration import compute_calibration, STRESS_THRESHOLD

    fp32_s, y = _scores(160, hedge=0.6, seed=0, flip=16)
    int8_s, _ = _scores(160, hedge=0.5, seed=3, flip=16)
    out = int8_calibration_drift(fp32_s, y, int8_s, y)

    # ECE on INT8 scores after applying the shipped fp32 temperature.
    scaled = temperature_scale(int8_s, out.shipped_temperature)
    expect = compute_calibration(scaled, y, fit_temperature=False).ece
    assert out.int8_ece_shipped_temp == pytest.approx(expect, abs=1e-9)


def test_temperature_transfers_is_bool():
    from model.int8_calibration_drift import int8_calibration_drift

    fp32_s, y = _scores(120, hedge=0.6, seed=0, flip=12)
    int8_s, _ = _scores(120, hedge=0.58, seed=1, flip=12)
    out = int8_calibration_drift(fp32_s, y, int8_s, y)
    assert isinstance(out.temperature_transfers, bool)


def test_verdict_mentions_int8():
    from model.int8_calibration_drift import int8_calibration_drift

    fp32_s, y = _scores(120, hedge=0.6, seed=0, flip=12)
    int8_s, _ = _scores(120, hedge=0.55, seed=1, flip=12)
    out = int8_calibration_drift(fp32_s, y, int8_s, y)
    assert "int8" in out.verdict.lower()
    assert len(out.verdict) > 20


def test_empty_rejected():
    from model.int8_calibration_drift import int8_calibration_drift

    with pytest.raises(ValueError):
        int8_calibration_drift(
            torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        )


def test_mismatched_lengths_rejected():
    from model.int8_calibration_drift import int8_calibration_drift

    fp32_s, y = _scores(100, hedge=0.6, seed=0, flip=10)
    int8_s, _ = _scores(80, hedge=0.5, seed=1, flip=8)
    with pytest.raises(ValueError):
        # int8 scores and labels disagree in length
        int8_calibration_drift(fp32_s, y, int8_s, y)


def test_to_dict_round_trips_json():
    from model.int8_calibration_drift import int8_calibration_drift

    fp32_s, y = _scores(120, hedge=0.6, seed=0, flip=12)
    int8_s, _ = _scores(120, hedge=0.5, seed=1, flip=12)
    out = int8_calibration_drift(fp32_s, y, int8_s, y)
    s = json.dumps(out.to_dict())
    back = json.loads(s)
    assert back["n"] == 120
    assert "ece_drift" in back
    assert back["fp32"]["label"] == "fp32"
    assert back["int8"]["label"] == "int8"
    assert "temperature_transfers" in back


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "int8_calibration_drift.json" in names


# --- heavy build path (tiny: trains + quantizes one small net) --------------

def test_build_trains_quantizes_and_writes_artifacts(tmp_path):
    from model.int8_calibration_drift import build_int8_calibration_drift

    out = build_int8_calibration_drift(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=8,
        snr_levels=(None, 0.0), out_dir=tmp_path,
    )
    assert out.n > 0
    assert out.fp32.label == "fp32" and out.int8.label == "int8"
    assert out.fp32.fitted_temperature > 0.0
    assert out.int8.fitted_temperature > 0.0
    assert (tmp_path / "int8_calibration_drift.json").is_file()
    assert (tmp_path / "int8_calibration_drift.md").is_file()
