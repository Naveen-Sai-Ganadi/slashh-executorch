"""A/B: global vs SNR-aware temperature (TDD, red first).

The per-SNR calibration breakdown (``model.calibration_snr``) showed the single
global ``DEFAULT_TEMPERATURE = 0.41`` under-corrects at the noisy floor: residual
ECE after temperature climbs from ~0.015 in the quiet to ~0.226 at -5 dB. This
A/B quantifies the *upper bound* on what an SNR-aware temperature could recover:
it pits the shipped global temperature against an oracle that knows each window's
true SNR and applies that SNR's own best temperature.

The oracle is not deployable (the device doesn't know the true SNR), but the gap
it recovers is the prize an on-device SNR estimator would chase. If the gap is
small, the global temperature is good enough and the estimator isn't worth
building; if it's large (especially at the floor), SNR-aware calibration is on
the table.

Host-only; no device, no AI Hub token. Pure reduction is unit-tested without
training; the heavy ``build_*`` path trains one tiny net.
"""

from __future__ import annotations

import json

import pytest
import torch


def _triple(snr, n, hedge, *, seed, flip):
    g = torch.Generator().manual_seed(seed)
    labels = (torch.arange(n) % 2).float()
    base = labels.clone()
    if flip:
        base[:flip] = 1.0 - base[:flip]
    scores = 0.5 + (base - 0.5) * hedge
    scores = (scores + 0.02 * torch.randn(n, generator=g)).clamp(0.01, 0.99)
    return (snr, scores, labels)


def _sweep():
    # clean is mildly under-confident; the floor is severely under-confident,
    # so an SNR-aware temperature should help most at the floor.
    return [
        _triple(None, 160, 0.85, seed=0, flip=4),
        _triple(10.0, 160, 0.6, seed=1, flip=10),
        _triple(-5.0, 160, 0.2, seed=2, flip=20),
    ]


def test_reports_one_comparison_per_snr():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    assert len(out.per_snr) == 3
    assert [c.snr_db for c in out.per_snr] == [None, 10.0, -5.0]


def test_global_temperature_defaults_to_constant():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab
    from model.confidence import DEFAULT_TEMPERATURE

    out = snr_aware_temperature_ab(_sweep())
    assert out.global_temperature == pytest.approx(DEFAULT_TEMPERATURE)


def test_global_ece_matches_temperature_scaled_calibration():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab
    from model.confidence import temperature_scale
    from model.calibration import compute_calibration

    sweep = _sweep()
    out = snr_aware_temperature_ab(sweep)
    for (snr, s, y), c in zip(sweep, out.per_snr):
        scaled = temperature_scale(s, out.global_temperature)
        expect = compute_calibration(scaled, y, fit_temperature=False).ece
        assert c.ece_global == pytest.approx(expect, abs=1e-9)


def test_oracle_not_worse_than_global_per_snr():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    # each SNR's own fitted temperature minimizes ECE on that SNR, so the
    # oracle never does worse than the single global temperature there.
    for c in out.per_snr:
        assert c.ece_oracle <= c.ece_global + 1e-6
        assert c.reduction >= -1e-6


def test_floor_is_lowest_snr():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    assert out.floor.snr_db == -5.0


def test_pooled_eces_present_and_ordered():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    assert out.pooled_ece_oracle <= out.pooled_ece_global + 1e-6
    assert out.pooled_reduction == pytest.approx(
        out.pooled_ece_global - out.pooled_ece_oracle, abs=1e-9
    )


def test_worth_it_is_bool():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    assert isinstance(out.worth_it, bool)


def test_verdict_nonempty():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    assert isinstance(out.verdict, str) and len(out.verdict) > 20


def test_empty_rejected():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    with pytest.raises(ValueError):
        snr_aware_temperature_ab([])


def test_to_dict_round_trips_json():
    from model.snr_aware_temperature_ab import snr_aware_temperature_ab

    out = snr_aware_temperature_ab(_sweep())
    s = json.dumps(out.to_dict())
    back = json.loads(s)
    assert len(back["per_snr"]) == 3
    assert "pooled_reduction" in back
    assert "worth_it" in back


def test_registered_in_benchmarks_index():
    from model.benchmarks_index import _ARTIFACTS

    names = [a[0] for a in _ARTIFACTS]
    assert "snr_aware_temperature_ab.json" in names


def test_build_trains_and_writes_artifacts(tmp_path):
    from model.snr_aware_temperature_ab import build_snr_aware_temperature_ab

    out = build_snr_aware_temperature_ab(
        seed=0, epochs=2, n_per_class=8, eval_n_per_class=8,
        snr_levels=(None, 0.0), out_dir=tmp_path,
    )
    assert len(out.per_snr) == 2
    assert (tmp_path / "snr_aware_temperature_ab.json").is_file()
    assert (tmp_path / "snr_aware_temperature_ab.md").is_file()
