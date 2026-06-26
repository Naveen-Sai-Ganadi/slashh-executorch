"""Smoke tests for runnable examples — keep the docs' demos from rotting.

The examples are part of the published surface (referenced from the README/docs),
so they must keep importing and running end to end on a clean checkout. This runs
the host-session demo's ``main()`` and asserts it produces the expected table —
cheap insurance that the public API the example shows still exists and composes.
Host-only; no device, no token.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout


def test_host_session_demo_runs_and_fires():
    from examples.host_session_demo import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    out = buf.getvalue()

    # The demo trains, prints the per-window table, and fires the calming cue.
    assert "trained:" in out
    assert "stressed" in out and "conf" in out
    assert "FIRE" in out, "sustained stress should fire the calming cue in the demo"
