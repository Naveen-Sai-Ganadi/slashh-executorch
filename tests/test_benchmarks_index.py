"""Tests for the benchmarks index generator (model/benchmarks_index.py).

The autonomy loop produces several JSON artifacts under docs/benchmarks/. This
generator rolls whatever is present into one README so a human (or the next
loop) gets a single legible entry point. It must be defensive: include only the
artifacts that exist, skip malformed ones, and never crash. Host-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.benchmarks_index import build_index


def _write(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj) + "\n")


def test_index_summarizes_present_artifacts(tmp_path: Path) -> None:
    _write(tmp_path / "robust_train.json", {
        "baseline": {"floor_db": 10.0, "points": []},
        "augmented": {"floor_db": None, "points": []},
    })
    _write(tmp_path / "robustness.json", {
        "floor_db": 20.0,
        "points": [{"snr_db": None, "accuracy": 1.0, "f1": 1.0, "n": 128}],
    })

    md = build_index(tmp_path)
    assert "# " in md                         # has a title
    assert "robust" in md.lower()             # mentions the robustness work
    # the README is written to disk
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "README.md").read_text() == md


def test_index_summarizes_int8_robustness(tmp_path: Path) -> None:
    _write(tmp_path / "int8_robustness.json", {
        "threshold": 0.8,
        "fp32_reliable_floor_db": 0.0,
        "int8_reliable_floor_db": 0.0,
        "preserves_floor": True,
        "acc_deltas": [{"snr_db": None, "delta": 0.0}],
        "fp32": {"floor_db": -5.0, "reliable_floor_db": 0.0, "points": []},
        "int8": {"floor_db": -5.0, "reliable_floor_db": 0.0, "points": []},
    })
    md = build_index(tmp_path)
    assert "INT8" in md
    assert "preserves the floor" in md


def test_index_summarizes_noise_colors(tmp_path: Path) -> None:
    _write(tmp_path / "noise_colors.json", {
        "threshold": 0.8,
        "reliable_floor_db": {"white": 0.0, "pink": 0.0, "brown": 10.0},
        "holds_across_colors": True,
        "curves": {},
    })
    md = build_index(tmp_path)
    assert "noise color" in md.lower()
    assert "holds across colors" in md


def test_index_summarizes_noise_failure_mode(tmp_path: Path) -> None:
    _write(tmp_path / "noise_failure_mode.json", {
        "threshold": 0.8,
        "dominant_failure": "misses_stress",
        "points": [],
    })
    md = build_index(tmp_path)
    assert "failure mode" in md.lower()
    assert "misses stress" in md


def test_index_summarizes_init_envelope(tmp_path: Path) -> None:
    _write(tmp_path / "init_envelope.json", {
        "threshold": 0.8,
        "envelope_db": 10.0,
        "n_inits": 5,
        "floors": [],
    })
    md = build_index(tmp_path)
    assert "envelope" in md.lower()
    assert "10 dB" in md


def test_index_summarizes_detector_robustness(tmp_path: Path) -> None:
    _write(tmp_path / "detector_robustness.json", {
        "window_count": 8,
        "detect_target": 0.8,
        "fa_tolerance": 0.2,
        "detection_floor_db": 10.0,
        "points": [],
    })
    md = build_index(tmp_path)
    assert "detector" in md.lower()
    assert "10 dB" in md


def test_index_summarizes_detector_noise_ab(tmp_path: Path) -> None:
    _write(tmp_path / "detector_noise_ab.json", {
        "fa_tolerance": 0.2,
        "recommended": {
            "config": {"label": "fast-attack", "stress_threshold": 0.5,
                       "release_threshold": 0.4, "ema_alpha": 1.0},
            "detection_floor_db": 0.0,
            "worst_false_alarm_rate": 0.1,
        },
        "results": [],
    })
    md = build_index(tmp_path)
    assert "fast-attack" in md
    assert "0 dB" in md


def test_index_detector_noise_ab_no_recommendation(tmp_path: Path) -> None:
    _write(tmp_path / "detector_noise_ab.json", {
        "fa_tolerance": 0.2, "recommended": None, "results": [],
    })
    md = build_index(tmp_path)
    assert "no config" in md.lower()


def test_missing_artifacts_are_skipped_not_fatal(tmp_path: Path) -> None:
    # only one artifact present; others absent
    _write(tmp_path / "benchmark.json", {
        "host": {"platform": "test", "torch": "x"},
        "results": [{"name": "base", "pte_bytes": 100000, "pte_latency_ms": 0.5}],
    })
    md = build_index(tmp_path)
    assert "base" in md
    # no crash despite the other four artifacts being absent
    assert isinstance(md, str) and len(md) > 0


def test_malformed_artifact_does_not_crash(tmp_path: Path) -> None:
    (tmp_path / "robustness.json").write_text("{ this is not valid json ")
    # should be skipped silently, index still builds
    md = build_index(tmp_path)
    assert isinstance(md, str)


def test_empty_dir_produces_a_placeholder(tmp_path: Path) -> None:
    md = build_index(tmp_path)
    assert isinstance(md, str) and len(md) > 0
    # mentions that nothing was found rather than erroring
    assert "no benchmark" in md.lower() or "nothing" in md.lower()
