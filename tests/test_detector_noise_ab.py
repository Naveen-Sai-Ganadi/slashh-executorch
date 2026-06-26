"""Tests for the noise-regime detector A/B (model/detector_noise_ab.py).

Two findings motivate this: the brittle model went *silent* under noise
(noise_failure_mode.py — recall collapses), while the very-aggressive production
model instead holds through its -5 dB envelope and the shipped detector's
end-to-end detection floor reaches -10 dB (detector_robustness.py). Both point at
the same hypothesis — in noisy regimes a *lower stress threshold / faster attack*
should recover detections at the noisy edge without paying much in false alarms,
precisely because the robust model isn't crying wolf inside its envelope. On the
production recipe this pans out: the lower-threshold config holds the floor to
-10 dB at a worst-case false-alarm rate inside budget, so the more sensitive
config wins.

This harness A/Bs detector configs over the SAME real model and the SAME noisy
traces (reusing the detector_robustness sweep per config) and recommends the
config that pushes the detection floor deepest into noise while keeping the
worst-case false-alarm rate within tolerance. Host-only; no device, no token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.detector_noise_ab import (
    DetectorConfig,
    DetectorNoiseABResult,
    detector_noise_ab,
)
from model.model import StressNet
from model.train import train


def _trained(seed: int = 0) -> StressNet:
    net = StressNet(channels=(4, 8, 16))
    train(data_dir=None, epochs=12, batch_size=16, lr=1e-3, n_per_class=64,
          seed=seed, model=net)
    return net.eval()


_CONFIGS = [
    DetectorConfig(label="default", stress_threshold=0.6, release_threshold=0.45, ema_alpha=0.4),
    DetectorConfig(label="fast-attack", stress_threshold=0.5, release_threshold=0.4, ema_alpha=1.0),
]


def test_one_result_per_config_with_floor_and_rates() -> None:
    model = _trained()
    out = detector_noise_ab(
        model, _CONFIGS, snr_levels=[None, 10.0, 0.0], n_traces=8,
        window_count=6, seed=1,
    )
    assert isinstance(out, DetectorNoiseABResult)
    assert [r.config.label for r in out.results] == ["default", "fast-attack"]
    for r in out.results:
        # detection_floor_db is None or a finite SNR
        assert r.detection_floor_db is None or isinstance(r.detection_floor_db, float)
        assert 0.0 <= r.worst_false_alarm_rate <= 1.0
        assert 0.0 <= r.mean_detect_rate <= 1.0


def test_recommends_a_config_within_fa_tolerance() -> None:
    # On clean audio the detector does not false-alarm, so a recommendation
    # exists and must be one of the swept configs, respecting the FA budget.
    model = _trained()
    out = detector_noise_ab(
        model, _CONFIGS, snr_levels=[None], n_traces=8,
        window_count=6, seed=2, fa_tolerance=0.25,
    )
    assert out.recommended is not None
    labels = {r.config.label for r in out.results}
    assert out.recommended.config.label in labels
    assert out.recommended.worst_false_alarm_rate <= 0.25


def test_no_safe_config_returns_none_recommendation() -> None:
    # Honest property: a *clean-trained* model false-alarms on calm audio once
    # white noise is added (it never learned noise = calm), so under a noisy
    # sweep no config stays within a tight FA budget — recommend nothing rather
    # than something unsafe. (The noise-augmented production model behaves the
    # opposite way — it goes silent — which is the point of build_*.)
    model = _trained()
    out = detector_noise_ab(
        model, _CONFIGS, snr_levels=[None, 10.0], n_traces=8,
        window_count=6, seed=2, fa_tolerance=0.2,
    )
    assert out.recommended is None
    assert all(r.worst_false_alarm_rate > 0.2 for r in out.results)


def test_deterministic_across_identical_calls() -> None:
    model = _trained()
    a = detector_noise_ab(model, _CONFIGS, snr_levels=[None, 0.0], n_traces=8, window_count=5, seed=3)
    b = detector_noise_ab(model, _CONFIGS, snr_levels=[None, 0.0], n_traces=8, window_count=5, seed=3)
    assert a.to_dict() == b.to_dict()


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    model = _trained()
    detector_noise_ab(
        model, _CONFIGS, snr_levels=[None, 10.0, 0.0], n_traces=8,
        window_count=6, seed=1, out_dir=tmp_path,
    )
    assert (tmp_path / "detector_noise_ab.json").is_file()
    assert (tmp_path / "detector_noise_ab.md").is_file()
    payload = json.loads((tmp_path / "detector_noise_ab.json").read_text())
    assert "results" in payload and "recommended" in payload
    md = (tmp_path / "detector_noise_ab.md").read_text()
    assert "detection floor" in md.lower()
