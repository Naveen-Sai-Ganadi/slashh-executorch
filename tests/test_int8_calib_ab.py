"""Tests for the INT8 calibration A/B (model/int8_calib_ab.py).

The shipped INT8 `.pte` is calibrated (PT2E observes activation ranges) on a
*clean* synthetic set, yet the model trains noise-augmented and runs on noisy
on-device audio. This harness quantizes the SAME trained model two ways — clean
calibration vs noise-augmented calibration — and asks whether matching the
calibration distribution to deployment preserves the noise-robustness floor any
better. Either answer is shippable; a 1,549-param net may well show no
difference. Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.int8_calib_ab import (
    CalibAbResult,
    CalibResult,
    int8_calib_ab,
)
from model.model import StressNet
from model.production import build_production_model
from model.train import train


def _trained(seed: int = 0) -> StressNet:
    net = build_production_model()
    train(data_dir=None, epochs=10, batch_size=16, lr=1e-3, n_per_class=48,
          seed=seed, model=net)
    return net.eval()


def _floor_key(v):
    # deeper (more negative) SNR floor is better; None = clean-only = worst.
    return float("inf") if v is None else v


def test_two_results_clean_and_noise_aware(tmp_path: Path) -> None:
    net = _trained()
    out = int8_calib_ab(
        net, snr_levels=[None, 10.0, 0.0], calib_n=12, n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    assert isinstance(out, CalibAbResult)
    assert [r.label for r in out.results] == ["clean", "noise-aware"]
    for r in out.results:
        assert isinstance(r, CalibResult)
        # each calibration produces a real INT8 curve over the same grid
        assert [p.snr_db for p in r.int8.points] == [None, 10.0, 0.0]
        assert r.reliable_floor_db is None or isinstance(r.reliable_floor_db, float)
        assert 0.0 <= r.mean_abs_delta <= 1.0
        assert len(r.acc_deltas) == 3
        for _snr, d in r.acc_deltas:
            assert -1.0 <= d <= 1.0


def test_fp32_baseline_shared_and_clean_tracks(tmp_path: Path) -> None:
    # Both calibrations are compared against the SAME fp32 curve, and on clean
    # audio INT8 must track fp32 closely regardless of calibration set.
    net = _trained()
    out = int8_calib_ab(
        net, snr_levels=[None, 10.0], calib_n=12, n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    fp_clean = {p.snr_db: p.accuracy for p in out.fp32.points}[None]
    assert fp_clean >= 0.8
    for r in out.results:
        q_clean = {p.snr_db: p.accuracy for p in r.int8.points}[None]
        assert abs(q_clean - fp_clean) <= 0.1


def test_verdict_is_one_of_three(tmp_path: Path) -> None:
    net = _trained()
    out = int8_calib_ab(
        net, snr_levels=[None, 10.0, 0.0], calib_n=12, n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    assert out.verdict in {"better", "same", "worse"}
    # verdict must agree with the actual floors it compares
    nf = _floor_key(out.noise_aware.reliable_floor_db)
    cf = _floor_key(out.clean.reliable_floor_db)
    expected = "better" if nf < cf else "same" if nf == cf else "worse"
    assert out.verdict == expected


def test_recommended_has_deepest_floor(tmp_path: Path) -> None:
    net = _trained()
    out = int8_calib_ab(
        net, snr_levels=[None, 10.0, 0.0], calib_n=12, n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    assert out.recommended in out.results
    other = out.clean if out.recommended is out.noise_aware else out.noise_aware
    # recommended floor is at least as deep as the other's
    assert _floor_key(out.recommended.reliable_floor_db) <= _floor_key(other.reliable_floor_db)
    # ties go to clean (cheaper, the current default)
    if _floor_key(out.clean.reliable_floor_db) == _floor_key(out.noise_aware.reliable_floor_db):
        assert out.recommended is out.clean


def test_deterministic_across_identical_calls() -> None:
    net = _trained()
    a = int8_calib_ab(net, snr_levels=[None, 0.0], calib_n=12, n_per_class=24, seed=2)
    b = int8_calib_ab(net, snr_levels=[None, 0.0], calib_n=12, n_per_class=24, seed=2)
    assert a.to_dict() == b.to_dict()


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    net = _trained()
    int8_calib_ab(
        net, snr_levels=[None, 10.0, 0.0], calib_n=12, n_per_class=24, seed=1,
        out_dir=tmp_path,
    )
    assert (tmp_path / "int8_calib_ab.json").is_file()
    assert (tmp_path / "int8_calib_ab.md").is_file()
    payload = json.loads((tmp_path / "int8_calib_ab.json").read_text())
    assert "results" in payload and "verdict" in payload and "recommended" in payload
    assert {r["label"] for r in payload["results"]} == {"clean", "noise-aware"}
    md = (tmp_path / "int8_calib_ab.md").read_text()
    assert "calibration" in md.lower()
    # the verdict line names which calibration to ship
    assert "recommend" in md.lower()
