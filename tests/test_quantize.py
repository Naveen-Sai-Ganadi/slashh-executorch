"""Tests for the host INT8 (PT2E + XNNPACK) quantized export path.

This is the on-host optimization variant: post-training static quantization via
the XNNPACK quantizer, lowered to a delegated ``.pte``. It does NOT use Qualcomm
AI Hub or the live token (that is the gated M5) — it runs entirely on the host
and produces a smaller program that still executes through the ExecuTorch runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from model.data import synthetic_dataset
from model.export_executorch import export_quantized_to_pte, export_to_pte
from model.model import StressNet, example_input
from model.run_pte import run_pte


def _calibration(n: int = 16) -> torch.Tensor:
    x, _ = synthetic_dataset(n, seed=1000)
    return x


def test_quantized_export_produces_runnable_pte() -> None:
    torch.manual_seed(0)
    model = StressNet().eval()
    buffer = export_quantized_to_pte(model, _calibration())

    assert isinstance(buffer, (bytes, bytearray))
    assert len(buffer) > 1000  # a real serialized program

    with tempfile.TemporaryDirectory() as td:
        pte = Path(td) / "q.pte"
        pte.write_bytes(buffer)
        out = run_pte(pte, example_input())
    assert tuple(out.shape) == (1, 1)
    score = float(out.flatten()[0])
    assert 0.0 <= score <= 1.0  # still a valid probability


def test_int8_is_smaller_than_fp32() -> None:
    torch.manual_seed(0)
    model = StressNet().eval()
    q_bytes = len(export_quantized_to_pte(model, _calibration()))
    fp32_bytes = len(export_to_pte(model=StressNet().eval()))
    # INT8 weights shrink the program meaningfully (~3x on this net)
    assert q_bytes < fp32_bytes


def test_quantized_benchmark_variant_flagged() -> None:
    from model.benchmark import benchmark_variant

    res = benchmark_variant(
        "base-int8", channels=(16, 32, 64), quantize=True,
        warmup=0, iters=2, eval_n=8,
    )
    assert res.quantized is True
    assert res.pte_latency_ms > 0
    assert res.pte_bytes > 1000
    # the fp32 variant of the same width is larger
    fp32 = benchmark_variant(
        "base", channels=(16, 32, 64), quantize=False, warmup=0, iters=2, eval_n=8
    )
    assert res.pte_bytes < fp32.pte_bytes
