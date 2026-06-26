"""Tests for the end-to-end latency + Real-Time Factor benchmark (model/latency_rtf.py).

The existing host benchmark (`model/benchmark.py`) times only the model forward
pass on a *precomputed* feature tensor. But the always-on product claim is
"real-time, offline, on-device": every HOP_SECONDS a fresh 3.0 s window must be
turned into a score before the next hop arrives, and that budget includes the
log-mel feature extraction, not just inference. This harness times the full
`raw waveform -> log-mel -> score` chain per window and reports the Real-Time
Factor against both the window length and the streaming hop deadline.

Timing is machine-dependent, so these tests assert STRUCTURE and INVARIANTS
(stage decomposition, the exact RTF relations, positivity/ordering), plus the
one safe absolute claim: a 1,549-param net on a dev host clears the 1.0 s hop
budget with vast margin. Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.audio_config import HOP_SECONDS, WINDOW_SECONDS
from model.export_executorch import export_to_pte
from model.latency_rtf import (
    LatencyResult,
    StageTiming,
    measure_latency,
)
from model.production import build_production_model


def _model():
    # Latency is weight-independent, so an untrained net is fine and keeps the
    # test fast. eval() to match the deployed forward path.
    return build_production_model().eval()


def test_eager_result_shape_and_stages() -> None:
    out = measure_latency(model=_model(), iters=8, warmup=2, seed=0)
    assert isinstance(out, LatencyResult)
    assert out.backend == "eager"
    assert out.iters == 8
    assert [s.name for s in out.stages] == ["feature_extraction", "inference"]
    for s in out.stages:
        assert isinstance(s, StageTiming)
        # every timing is a positive, sanely-ordered millisecond figure
        assert s.min_ms > 0.0
        assert s.p50_ms >= s.min_ms
        assert s.p95_ms >= s.p50_ms
        assert s.mean_ms >= s.min_ms


def test_total_is_sum_of_stage_means() -> None:
    out = measure_latency(model=_model(), iters=8, warmup=2, seed=1)
    fe = out.feature_extraction.mean_ms
    inf = out.inference.mean_ms
    assert abs(out.total_mean_ms - (fe + inf)) <= 1e-6


def test_rtf_relations_are_exact() -> None:
    out = measure_latency(model=_model(), iters=8, warmup=2, seed=2)
    assert out.window_seconds == WINDOW_SECONDS
    assert out.hop_seconds == HOP_SECONDS
    assert abs(out.rtf_window - (out.total_mean_ms / 1000.0) / WINDOW_SECONDS) <= 1e-9
    assert abs(out.rtf_hop - (out.total_mean_ms / 1000.0) / HOP_SECONDS) <= 1e-9
    # the hop budget is the tighter of the two deadlines
    assert out.rtf_hop >= out.rtf_window
    assert out.real_time == (out.rtf_hop < 1.0)


def test_tiny_net_is_real_time_on_host() -> None:
    # Sub-millisecond inference + a few-ms feature extract against a 1000 ms hop
    # budget: this must clear real-time with large margin on any dev host.
    out = measure_latency(model=_model(), iters=12, warmup=3, seed=3)
    assert out.real_time is True
    assert out.rtf_hop < 0.5


def test_requires_exactly_one_backend() -> None:
    import pytest

    with pytest.raises(ValueError):
        measure_latency(iters=4)
    with pytest.raises(ValueError):
        measure_latency(model=_model(), pte_bytes=b"x", iters=4)


def test_pte_backend_runs_and_labels() -> None:
    pte = export_to_pte(model=_model())
    out = measure_latency(pte_bytes=pte, iters=6, warmup=2, seed=4)
    assert out.backend == "pte"
    assert [s.name for s in out.stages] == ["feature_extraction", "inference"]
    assert out.inference.mean_ms > 0.0
    assert out.real_time is True


def test_to_dict_has_expected_keys() -> None:
    out = measure_latency(model=_model(), iters=6, warmup=2, seed=5)
    d = out.to_dict()
    for k in (
        "backend", "stages", "total_mean_ms", "total_p95_ms",
        "window_seconds", "hop_seconds", "rtf_window", "rtf_hop",
        "real_time", "iters",
    ):
        assert k in d
    assert {s["name"] for s in d["stages"]} == {"feature_extraction", "inference"}
    # round-trips through JSON
    assert json.loads(json.dumps(d)) == d


def test_backend_label_override() -> None:
    out = measure_latency(model=_model(), iters=4, warmup=1, seed=6, backend="eager-fp32")
    assert out.backend == "eager-fp32"
