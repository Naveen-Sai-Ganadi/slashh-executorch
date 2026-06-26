---
name: example-scaffold
description: Use to scaffold a new end-to-end example — a model that exports to a .pte and runs through the runtime — reusing the existing AOT + runtime stack, with a parity check wired in.
---

# Example Scaffold (export → run)

Recipe to stand up a new example under `examples/` that exercises the full path: a PyTorch model → exported `.pte` → run through the runtime, with a numerical-parity check.

## Shared principles
- Reuse the AOT stack (`torch.export` → `to_edge` → `to_executorch`) and the runtime/runner — never reimplement export or execution in the example.
- Offline-first: the example must run with a small, locally-defined or vendored model (no weight downloads in CI).
- Wire a **parity assertion** vs the eager-PyTorch reference so the example doubles as a UAT fixture.
- Add the example to the `examples/` index and to the CI matrix where it should gate.

## Layout
```
examples/<name>/
  model.py        # the nn.Module (or a tiny vendored one)
  export.py       # export → .pte  (entry point the uat-example skill calls)
  run.py          # load the .pte, run it, print/return outputs
  test_<name>.py  # pytest: export, run, assert parity within tolerance
  README.md       # what it demonstrates + how to run
```

## Steps
1. Define or vendor a small `nn.Module` in `model.py`.
2. In `export.py`, export it to `/tmp/<name>.pte` via the AOT flow the project uses.
3. In `run.py`, load and execute the `.pte` (C++ runner or pybindings) and capture outputs.
4. In `test_<name>.py`, assert runtime output matches the eager reference within tolerance (e.g. `max abs err < 1e-3`).
5. Register it in the `examples/` index; have `devops-engineer` add it to CI if it should gate.

## Return
Return the example path, the model exercised, how to run it, and the parity tolerance used.
