"""fp32 vs INT8 .pte on-disk footprint A/B (TDD, red first).

Every existing INT8 artifact measures *accuracy/calibration* under quantization
(int8_robustness, int8_calib_ab, int8_calibration_drift). None measures the
thing quantization is actually for on-device: the program *shrinks*. This A/B
exports the SAME trained production model two ways — fp32 XNNPACK vs INT8
PT2E — and compares the serialized .pte byte size, the concrete cost behind
"INT8 on the NPU".

Honest angle: a 1,549-param net is tiny, so the .pte's fixed runtime/header
overhead is a large fraction of the file. The realized compression may fall
well short of the textbook 4x (fp32 32-bit -> int8 8-bit weights) — if so,
that's the documented finding, not a number to dress up.

Host-only; no device, no AI Hub token. Pure reduction is unit-tested without
exporting; the heavy build path trains + exports one tiny net.
"""

from __future__ import annotations

import json

import pytest


# --- pure reduction logic (no export) -------------------------------------

def test_compression_ratio_and_savings():
    from model.pte_footprint import pte_footprint

    out = pte_footprint(fp32_bytes=400_000, int8_bytes=100_000, n_params=1549)
    assert out.compression_ratio == pytest.approx(4.0)
    assert out.savings_pct == pytest.approx(75.0)


def test_bytes_per_param():
    from model.pte_footprint import pte_footprint

    out = pte_footprint(fp32_bytes=400_000, int8_bytes=100_000, n_params=1000)
    assert out.fp32_bytes_per_param == pytest.approx(400.0)
    assert out.int8_bytes_per_param == pytest.approx(100.0)


def test_pct_of_theoretical():
    from model.pte_footprint import pte_footprint

    # 2x realized vs the 4x textbook ceiling for fp32->int8 weights.
    out = pte_footprint(fp32_bytes=200_000, int8_bytes=100_000, n_params=1549)
    assert out.compression_ratio == pytest.approx(2.0)
    assert out.pct_of_theoretical == pytest.approx(50.0)


def test_material_shrink_is_bool():
    from model.pte_footprint import pte_footprint

    out = pte_footprint(fp32_bytes=400_000, int8_bytes=100_000, n_params=1549)
    assert isinstance(out.material_shrink, bool)
    assert out.material_shrink is True


def test_int8_not_smaller_is_reported_not_error():
    from model.pte_footprint import pte_footprint

    # for a tiny model, fixed overhead can leave INT8 no smaller — a real,
    # honest finding, not an exception.
    out = pte_footprint(fp32_bytes=100_000, int8_bytes=110_000, n_params=1549)
    assert out.compression_ratio < 1.0
    assert out.savings_pct < 0.0
    assert out.material_shrink is False
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_nonpositive_rejected():
    from model.pte_footprint import pte_footprint

    with pytest.raises(ValueError):
        pte_footprint(fp32_bytes=0, int8_bytes=100, n_params=1549)
    with pytest.raises(ValueError):
        pte_footprint(fp32_bytes=100, int8_bytes=100, n_params=0)


def test_verdict_nonempty():
    from model.pte_footprint import pte_footprint

    out = pte_footprint(fp32_bytes=400_000, int8_bytes=100_000, n_params=1549)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.pte_footprint import pte_footprint

    out = pte_footprint(fp32_bytes=400_000, int8_bytes=100_000, n_params=1549)
    back = json.loads(json.dumps(out.to_dict()))
    assert back["fp32_bytes"] == 400_000
    assert back["int8_bytes"] == 100_000
    assert "compression_ratio" in back
    assert "savings_pct" in back
    assert "material_shrink" in back


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "pte_footprint.json" in names


# --- heavy build path (trains + exports one small net) ----------------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.pte_footprint import build_pte_footprint

    out = build_pte_footprint(
        seed=0, epochs=2, n_per_class=8, calib_n=6, out_dir=tmp_path,
    )
    assert out.fp32_bytes > 0
    assert out.int8_bytes > 0
    assert out.n_params > 0
    assert (tmp_path / "pte_footprint.json").is_file()
    assert (tmp_path / "pte_footprint.md").is_file()
