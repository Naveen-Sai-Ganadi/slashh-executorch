"""Per-channel vs per-tensor INT8 granularity A/B (TDD, red first).

The shipped INT8 ``.pte`` quantizes with ``is_per_channel=True`` — a separate
scale/zero-point per output channel. The simpler, cheaper alternative is
*per-tensor*: one scale/zero-point for the whole weight tensor. Per-tensor is
smaller (no per-channel scale vectors) and is sometimes the only mode a fixed-
function NPU backend supports — but it can lose accuracy when a layer's weight
distribution varies channel-to-channel.

Every existing INT8 artifact (int8_robustness, int8_calib_ab,
int8_calibration_drift, pte_footprint) holds the *granularity* fixed at
per-channel and varies something else (calibration data, noise, drift, size).
None asks the granularity question itself: on this 1,549-param net, does
per-tensor cost accuracy/floor, and how much smaller is it?

Honest angle: for a tiny net the two may be indistinguishable in accuracy, in
which case per-tensor is the cheaper equivalent — a real, documented finding,
not a number to dress up.

Host-only; no device, no AI Hub token. The pure reduction is unit-tested
without exporting; the heavy build path trains + exports one tiny net twice.
"""

from __future__ import annotations

import json

import pytest


# --- pure reduction logic (no export) -------------------------------------

def _curve(*accs):
    """Build (snr_db, acc_per_channel, acc_per_tensor) triples from accuracy
    pairs, clean->noisy over the standard SNR ladder."""
    snrs = [None, 20.0, 10.0, 0.0, -5.0]
    return [(snrs[i], pc, pt) for i, (pc, pt) in enumerate(accs)]


def test_reports_one_point_per_snr():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((1.0, 1.0), (0.95, 0.95), (0.9, 0.9))
    out = int8_granularity_ab(pts, per_channel_bytes=13000, per_tensor_bytes=12500)
    assert len(out.points) == 3
    assert [p.snr_db for p in out.points] == [None, 20.0, 10.0]


def test_delta_is_per_tensor_minus_per_channel():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((0.9, 0.8), (0.85, 0.9))
    out = int8_granularity_ab(pts, per_channel_bytes=13000, per_tensor_bytes=12500)
    assert out.points[0].delta == pytest.approx(-0.1)
    assert out.points[1].delta == pytest.approx(0.05)


def test_floors_computed_contiguously():
    from model.int8_granularity_ab import int8_granularity_ab

    # per-channel holds to 0 dB; per-tensor collapses one step earlier (10 dB).
    pts = _curve((1.0, 1.0), (0.95, 0.95), (0.9, 0.9), (0.85, 0.5), (0.6, 0.5))
    out = int8_granularity_ab(pts, per_channel_bytes=13000, per_tensor_bytes=12500)
    assert out.floor_per_channel == 0.0
    assert out.floor_per_tensor == 10.0
    assert out.per_tensor_holds_floor is False


def test_identical_curves_hold_floor_and_no_material_gap():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((1.0, 1.0), (0.95, 0.95), (0.9, 0.9), (0.85, 0.85))
    out = int8_granularity_ab(pts, per_channel_bytes=13000, per_tensor_bytes=12500)
    assert out.per_tensor_holds_floor is True
    assert out.material_accuracy_gap is False
    assert out.worth_per_channel is False


def test_byte_savings_pct():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((0.9, 0.9))
    out = int8_granularity_ab(pts, per_channel_bytes=1000, per_tensor_bytes=900)
    assert out.byte_savings_pct == pytest.approx(10.0)


def test_max_abs_delta_and_material_gap_is_bool():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((0.9, 0.9), (0.9, 0.78))  # 0.12 gap at 20 dB
    out = int8_granularity_ab(pts, per_channel_bytes=1000, per_tensor_bytes=900)
    assert out.max_abs_delta == pytest.approx(0.12)
    assert isinstance(out.material_accuracy_gap, bool)
    assert out.material_accuracy_gap is True


def test_empty_rejected():
    from model.int8_granularity_ab import int8_granularity_ab

    with pytest.raises(ValueError):
        int8_granularity_ab([], per_channel_bytes=1000, per_tensor_bytes=900)


def test_nonpositive_bytes_rejected():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((0.9, 0.9))
    with pytest.raises(ValueError):
        int8_granularity_ab(pts, per_channel_bytes=0, per_tensor_bytes=900)
    with pytest.raises(ValueError):
        int8_granularity_ab(pts, per_channel_bytes=1000, per_tensor_bytes=-1)


def test_verdict_nonempty():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((0.9, 0.9), (0.85, 0.85))
    out = int8_granularity_ab(pts, per_channel_bytes=1000, per_tensor_bytes=900)
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_to_dict_round_trips_json():
    from model.int8_granularity_ab import int8_granularity_ab

    pts = _curve((1.0, 1.0), (0.9, 0.85))
    out = int8_granularity_ab(pts, per_channel_bytes=13000, per_tensor_bytes=12500)
    back = json.loads(json.dumps(out.to_dict()))
    assert len(back["points"]) == 2
    assert "byte_savings_pct" in back
    assert "per_tensor_holds_floor" in back
    assert "worth_per_channel" in back


# --- index registration ----------------------------------------------------

def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "int8_granularity_ab.json" in names


# --- heavy build path (trains + exports one small net twice) ----------------

def test_build_trains_and_writes_artifacts(tmp_path):
    from model.int8_granularity_ab import build_int8_granularity_ab

    out = build_int8_granularity_ab(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=8, calib_n=6,
        snr_levels=(None, 0.0), out_dir=tmp_path,
    )
    assert len(out.points) == 2
    assert out.per_channel_bytes > 0
    assert out.per_tensor_bytes > 0
    assert (tmp_path / "int8_granularity_ab.json").is_file()
    assert (tmp_path / "int8_granularity_ab.md").is_file()
