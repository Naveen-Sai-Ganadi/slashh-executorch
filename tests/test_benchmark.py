"""Tests for the host benchmark + A/B harness (model/benchmark.py).

The harness measures, on the host, the things M11 needs as evidence:
latency, model size, and accuracy — for one or more StressNet variants — and
emits a machine-readable record plus a human-readable markdown table. None of
this touches the live AI Hub token or a device; it is the CPU/XNNPACK baseline
the NPU numbers later get compared against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from model.benchmark import (
    BenchmarkResult,
    benchmark_variant,
    run_suite,
    to_markdown,
)
from model.model import StressNet


def test_benchmark_variant_reports_latency_size_accuracy() -> None:
    res = benchmark_variant("tiny", channels=(8, 16, 32), warmup=1, iters=3, eval_n=16)

    assert isinstance(res, BenchmarkResult)
    assert res.name == "tiny"
    # latency: eager always present; pte present because export should succeed
    assert res.eager_latency_ms > 0
    assert res.pte_latency_ms > 0
    # size: the .pte is a real serialized program of non-trivial size
    assert res.pte_bytes > 1000
    assert res.pt_bytes > 0
    # accuracy in [0, 1]; parity error is tiny (XNNPACK vs eager fp32)
    assert 0.0 <= res.accuracy <= 1.0
    assert res.pte_max_abs_err < 1e-2


def test_benchmark_variant_is_serializable() -> None:
    res = benchmark_variant("tiny", channels=(8, 16, 32), warmup=0, iters=2, eval_n=8)
    d = res.to_dict()
    # round-trips through json without loss of the key fields
    again = json.loads(json.dumps(d))
    assert again["name"] == "tiny"
    assert again["pte_bytes"] == res.pte_bytes
    assert "eager_latency_ms" in again and "accuracy" in again


def test_run_suite_compares_two_variants_and_writes_artifacts(tmp_path: Path) -> None:
    variants = {
        "a": (8, 16, 32),
        "b": (16, 32, 64),
    }
    out = run_suite(variants, out_dir=tmp_path, warmup=0, iters=2, eval_n=8)

    # one result per variant, A/B comparable
    assert set(r.name for r in out.results) == {"a", "b"}
    # the larger variant should be at least as large on disk
    by_name = {r.name: r for r in out.results}
    assert by_name["b"].pte_bytes >= by_name["a"].pte_bytes

    # artifacts written: a json record and a markdown table
    json_path = tmp_path / "benchmark.json"
    md_path = tmp_path / "benchmark.md"
    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text())
    assert len(payload["results"]) == 2
    assert "host" in payload  # environment provenance recorded


def test_to_markdown_has_a_row_per_variant() -> None:
    results = [
        benchmark_variant("a", channels=(8, 16, 32), warmup=0, iters=1, eval_n=8),
        benchmark_variant("b", channels=(16, 32, 64), warmup=0, iters=1, eval_n=8),
    ]
    md = to_markdown(results)
    assert "| variant " in md or "| name " in md
    # a data row for each variant name
    assert "a" in md and "b" in md
    # latency and size columns are present in the header
    low = md.lower()
    assert "latency" in low and ("size" in low or "bytes" in low)


def test_default_suite_runs_with_a_single_variant(tmp_path: Path) -> None:
    # smallest possible real run: one variant, exercised end to end
    out = run_suite({"default": (16, 32, 64)}, out_dir=tmp_path, warmup=0, iters=2, eval_n=8)
    assert len(out.results) == 1
    assert out.results[0].pte_latency_ms > 0
