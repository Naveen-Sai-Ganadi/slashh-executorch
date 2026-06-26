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
