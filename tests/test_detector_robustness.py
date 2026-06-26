"""Tests for end-to-end detector robustness (model/detector_robustness.py).

model/robustness.py and model/noise_failure_mode.py measure the *raw model*
under noise. But the product ships the detector: EMA smoothing + dual-threshold
hysteresis over a *stream* of windows (model/detector.py). Smoothing trades
latency for stability, so the end-to-end decision can hold where a single
window wouldn't — or it can simply inherit the model's collapse. This harness
streams multi-window noisy traces through HostStressPipeline and measures, per
SNR, the detection rate on stressed traces, the false-alarm rate on calm
traces, and the median windows-to-latch. Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.detector_robustness import (
    DetectorRobustnessResult,
    detector_robustness,
)
from model.model import StressNet
from model.train import train


def _trained(seed: int = 0) -> StressNet:
    net = StressNet(channels=(4, 8, 16))
    train(data_dir=None, epochs=12, batch_size=16, lr=1e-3, n_per_class=64,
          seed=seed, model=net)
    return net.eval()


def test_points_have_rates_in_range_and_one_per_snr() -> None:
    model = _trained()
    out = detector_robustness(
        model, snr_levels=[None, 10.0, 0.0], n_traces=8, window_count=6, seed=1,
    )
    assert isinstance(out, DetectorRobustnessResult)
    assert [p.snr_db for p in out.points] == [None, 10.0, 0.0]
    for p in out.points:
        assert 0.0 <= p.detect_rate <= 1.0
        assert 0.0 <= p.false_alarm_rate <= 1.0
        assert p.n_traces == 8
        # latency is None (nothing detected) or a positive window count <= T
        if p.median_latency_windows is not None:
            assert 1 <= p.median_latency_windows <= 6


def test_clean_audio_detects_stress_and_does_not_false_alarm() -> None:
    model = _trained()
    out = detector_robustness(
        model, snr_levels=[None], n_traces=12, window_count=6, seed=2,
    )
    clean = out.points[0]
    # a usable detector fires on clean stressed speech and stays quiet on calm
    assert clean.detect_rate >= 0.7
    assert clean.false_alarm_rate <= 0.3


def test_deterministic_across_identical_calls() -> None:
    model = _trained()
    a = detector_robustness(model, snr_levels=[None, 0.0], n_traces=8, window_count=5, seed=3)
    b = detector_robustness(model, snr_levels=[None, 0.0], n_traces=8, window_count=5, seed=3)
    assert a.to_dict() == b.to_dict()


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    model = _trained()
    detector_robustness(
        model, snr_levels=[None, 10.0, 0.0], n_traces=8, window_count=6, seed=1,
        out_dir=tmp_path,
    )
    assert (tmp_path / "detector_robustness.json").is_file()
    assert (tmp_path / "detector_robustness.md").is_file()
    payload = json.loads((tmp_path / "detector_robustness.json").read_text())
    assert "points" in payload
    md = (tmp_path / "detector_robustness.md").read_text()
    assert "detect" in md.lower() and "false alarm" in md.lower()
