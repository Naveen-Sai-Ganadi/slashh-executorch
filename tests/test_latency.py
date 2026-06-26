"""Real-time guarantee for the host stress pipeline (model/pipeline.py).

The README promises stress scoring "in real time" — each ~3-second window must
be processed in much less than its own duration, or the on-device meter falls
behind the mic. This guards that claim against a hot-path regression (e.g. a
heavy op sneaking into feature extraction or the model).

Margins are enormous (spike: ~0.85 ms to process a 3.0 s window, RTF ~0.0003),
so the assertions below sit ~300x inside the budget and tolerate a slow or
loaded CI runner. Timing-based but not flaky: we warm up, take the median, and
assert against a generous bound. Host-only; no device, no AI Hub, no token.
"""

from __future__ import annotations

import torch

from model.audio_config import SAMPLE_RATE, WINDOW_SAMPLES
from model.data import _synth_waveform
from model.pipeline import HostStressPipeline, realtime_factor
from model.production import train_production


def _pipe() -> HostStressPipeline:
    torch.manual_seed(0)
    model, _ = train_production(epochs=12, n_per_class=96, seed=0)
    return HostStressPipeline(model)


def _windows(n: int = 20) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return [_synth_waveform(bool(i % 2), gen) for i in range(n)]


def test_pipeline_runs_in_real_time() -> None:
    report = realtime_factor(_pipe(), _windows())

    # the window is the real 3-second analysis frame
    assert abs(report["window_s"] - WINDOW_SAMPLES / SAMPLE_RATE) < 1e-9

    # median processing is far inside the window budget (~300x margin to 0.1)
    assert report["median_rtf"] < 0.1
    # even the worst single window stays comfortably real-time (absorbs a hiccup)
    assert report["max_rtf"] < 0.25
    # sanity: timings are positive and max >= median
    assert 0.0 < report["median_ms"] <= report["max_ms"]


def test_realtime_factor_reports_each_window() -> None:
    windows = _windows(12)
    report = realtime_factor(_pipe(), windows, warmup=3)
    # one timing per measured (non-warmup) window
    assert len(report["per_window_ms"]) == len(windows)
    assert all(t > 0 for t in report["per_window_ms"])
