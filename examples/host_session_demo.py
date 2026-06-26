"""Runnable demo of the end-to-end host loop (HostStressSession).

Streams a sequence of synthetic windows through the full host chain —
waveform → log-mel → StressNet → detector → calibrated confidence + calming-cue
decision — and prints, per window, the gated decision, the (non-gating)
confidence read-out, and whether the calming intervention fired.

This mirrors what the on-device loop does each ~1 s hop; everything here is
host-only, offline, and uses no AI Hub token. A small StressNet is trained for a
few epochs on the synthetic proxy data so the stress path actually crosses the
gate and the cue fires — the point is to exercise the whole chain end to end on a
real (if tiny) model. Swap in `model.production.train_production` for the shipped
recipe.

Run:
    PYTHONPATH=. .venv/bin/python -m examples.host_session_demo
"""

from __future__ import annotations

import torch

from model.data import _synth_waveform
from model.intervention import InterventionConfig, InterventionPolicy
from model.model import StressNet
from model.session import HostStressSession
from model.train import train


def main() -> None:
    torch.manual_seed(0)

    # Train a small production-width net for a few epochs on the synthetic proxy
    # so the "stressed" windows below actually score above the gate.
    model, meta = train(
        data_dir=None, epochs=12, batch_size=32, lr=1e-3, n_per_class=96, seed=0,
        model=StressNet(channels=(4, 8, 16)),
    )
    print(f"trained: val_acc={meta.get('val_acc', float('nan')):.3f}\n")

    # A short, more sensitive cue so the demo fires within a few windows.
    policy = InterventionPolicy(InterventionConfig(sustain_windows=2, cooldown_windows=4))
    session = HostStressSession(model, intervention=policy)

    # A deterministic stream: a few unvoiced (silent) windows, then voiced
    # *stressed* windows, so we can watch the gate latch and the cue fire.
    gen = torch.Generator().manual_seed(1)
    voiced_flags = [False, False, True, True, True, True, True, True, True, True]

    print(f"{'win':>3}  {'voiced':>6}  {'raw':>6}  {'stressed':>8}  {'conf':>5}  {'cue':>4}")
    print("-" * 44)
    for i, voiced in enumerate(voiced_flags):
        # Silent windows are gated out; voiced ones carry a stressed proxy signal.
        wave = _synth_waveform(stressed=True, gen=gen)
        out = session.process_window(wave, voiced=voiced)
        raw = "  --  " if out.state.raw_score is None else f"{out.state.raw_score:6.3f}"
        conf = "  -- " if out.confidence is None else f"{out.confidence.confidence:5.2f}"
        cue = "FIRE" if out.intervention.fire else "  · "
        print(f"{i:>3}  {str(voiced):>6}  {raw}  {str(out.state.stressed):>8}  {conf}  {cue}")

    print(
        "\nThe gate owns 'stressed'; 'conf' is a non-gating calibrated read-out; "
        "'cue' fires only on sustained stress and then rate-limits (no nagging)."
    )


if __name__ == "__main__":
    main()
