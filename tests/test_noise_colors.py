"""Tests for colored-noise robustness (model/noise_colors.py + robustness color).

The base robustness sweep injects *white* Gaussian noise. Real acoustic noise is
rarely white: traffic, fans, and HVAC skew toward pink/brown (more energy in low
frequencies). This checks the noise generators are well-behaved and that the
robustness sweep can run with any noise color, then certifies the floor across
colors. Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model.model import StressNet
from model.noise_colors import noise_color_robustness, NoiseColorResult
from model.robustness import noisy_synthetic_dataset
from model.train import train


def _trained(seed: int = 0) -> StressNet:
    net = StressNet()
    train(data_dir=None, epochs=10, batch_size=16, lr=1e-3, n_per_class=48,
          seed=seed, model=net)
    return net.eval()


def test_colored_noise_dataset_is_deterministic_and_distinct() -> None:
    # same seed + color reproduces; different colors differ; all finite, right shape
    x_pink = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=10.0, noise_color="pink")[0]
    again = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=10.0, noise_color="pink")[0]
    assert torch.allclose(x_pink, again)
    assert torch.isfinite(x_pink).all()
    assert tuple(x_pink.shape) == (16, 1, 64, 301)

    x_white = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=10.0, noise_color="white")[0]
    x_brown = noisy_synthetic_dataset(n_per_class=8, seed=0, snr_db=10.0, noise_color="brown")[0]
    assert not torch.allclose(x_pink, x_white)
    assert not torch.allclose(x_pink, x_brown)


def test_white_default_is_unchanged() -> None:
    # the default noise_color reproduces the original white-noise draw exactly,
    # so existing white-noise records/tests stay byte-stable.
    default = noisy_synthetic_dataset(n_per_class=8, seed=1, snr_db=0.0)[0]
    explicit_white = noisy_synthetic_dataset(n_per_class=8, seed=1, snr_db=0.0, noise_color="white")[0]
    assert torch.allclose(default, explicit_white)


def test_colored_noise_honors_snr() -> None:
    # pink noise at a given SNR must produce roughly that SNR on the waveform:
    # measured 10*log10(sig_power / noise_power) within a couple dB of requested.
    from model.robustness import _add_noise
    from model.data import _synth_waveform

    gen = torch.Generator().manual_seed(3)
    wave = _synth_waveform(True, gen)
    for snr_db in (10.0, 0.0):
        ngen = torch.Generator().manual_seed(7)
        noisy = _add_noise(wave, snr_db, ngen, color="pink")
        noise = noisy - wave
        meas = 10.0 * torch.log10(wave.pow(2).mean() / noise.pow(2).mean().clamp(min=1e-12))
        assert abs(float(meas) - snr_db) < 2.0


def test_noise_color_robustness_sweeps_all_colors(tmp_path: Path) -> None:
    model = _trained(seed=0)
    out = noise_color_robustness(
        model, colors=["white", "pink", "brown"],
        snr_levels=[None, 10.0, 0.0], n_per_class=24, seed=1, out_dir=tmp_path,
    )
    assert isinstance(out, NoiseColorResult)
    assert set(out.curves) == {"white", "pink", "brown"}
    for color, curve in out.curves.items():
        assert [p.snr_db for p in curve.points] == [None, 10.0, 0.0]
        # clean point is identical regardless of color (no noise injected)
        clean = {p.snr_db: p.accuracy for p in curve.points}[None]
        assert clean > 0.7

    # artifacts written, and the per-color floors are recorded
    assert (tmp_path / "noise_colors.json").is_file()
    assert (tmp_path / "noise_colors.md").is_file()
    payload = json.loads((tmp_path / "noise_colors.json").read_text())
    assert set(payload["reliable_floor_db"]) == {"white", "pink", "brown"}


def test_noise_color_markdown_reports_per_color_floor(tmp_path: Path) -> None:
    model = _trained(seed=0)
    noise_color_robustness(
        model, colors=["white", "pink"], snr_levels=[None, 10.0],
        n_per_class=24, seed=1, out_dir=tmp_path,
    )
    md = (tmp_path / "noise_colors.md").read_text()
    assert "white" in md and "pink" in md
