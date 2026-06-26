"""Run a ``.pte`` through the ExecuTorch runtime (host pybindings).

    python -m model.run_pte --model assets/stress_model.pte

This is the host-side stand-in for the on-device runtime: it loads the program
and executes ``forward`` exactly as the Android runtime will. Used by the parity
test (model/../tests) to prove the exported program matches eager PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from executorch.runtime import Runtime


def run_pte(model_path: str | Path, inputs: torch.Tensor) -> torch.Tensor:
    """Execute ``forward`` on a loaded ``.pte`` and return the output tensor."""
    runtime = Runtime.get()
    program = runtime.load_program(Path(model_path))
    method = program.load_method("forward")
    outputs = method.execute([inputs])
    return outputs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a StressNet .pte")
    ap.add_argument("--model", "-m", default="assets/stress_model.pte")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from .model import example_input

    torch.manual_seed(args.seed)
    out = run_pte(args.model, example_input())
    print(f"stress score: {out.flatten().tolist()}")


if __name__ == "__main__":
    main()
