"""Tests for the front-end batching throughput A/B (feature_batching_bench).

The benchmark's *numbers* are machine-dependent, so the assertions pin its
invariants and its parity claim, not wall-clock thresholds: the two paths must
agree to tolerance, the result must serialize, and the artifact must be written.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.feature_batching_bench import (
    BatchingResult,
    build_feature_batching,
    measure_batching,
    to_markdown,
)


def test_measure_reports_parity_and_positive_times():
    r = measure_batching(n=16, repeats=2, warmup=1, seed=0)
    assert isinstance(r, BatchingResult)
    assert r.n == 16
    assert r.loop_ms > 0.0 and r.batch_ms > 0.0
    assert r.speedup > 0.0
    # batched front-end must match the per-sample loop to tolerance
    assert r.parity_ok
    assert r.max_abs_diff <= r.parity_atol


def test_result_serializes():
    r = measure_batching(n=8, repeats=2, warmup=1, seed=1)
    d = r.to_dict()
    for k in ("n", "loop_ms", "batch_ms", "speedup", "max_abs_diff", "parity_ok"):
        assert k in d
    assert json.loads(json.dumps(d)) == d


def test_markdown_mentions_speedup_and_parity():
    r = measure_batching(n=8, repeats=2, warmup=1, seed=2)
    md = to_markdown(r).lower()
    assert "speedup" in md and "parity" in md


def test_build_writes_artifacts(tmp_path: Path):
    r = build_feature_batching(n=8, repeats=2, out_dir=tmp_path)
    assert (tmp_path / "feature_batching.json").is_file()
    assert (tmp_path / "feature_batching.md").is_file()
    assert r.parity_ok
