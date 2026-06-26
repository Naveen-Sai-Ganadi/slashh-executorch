---
name: uat-example
description: Use to run end-to-end UAT for the inference runtime — export a model to a .pte, run it through the runtime, assert numerical parity vs eager PyTorch, and certify the UAT half of the ship sentinel.
---

# UAT by Example (export → run → parity)

Exercise the real path a developer takes — export a model and run it through the actual runtime — and certify the result. Used by `uat-runner`.

## Steps

1. **Export** the model under test to a `.pte` with the AOT stack (the export entry point / `to_edge` → `to_executorch` flow the charge specifies):
   ```sh
   .venv/bin/python -m <export_script> --model <name> --output /tmp/<name>.pte
   ```
   Capture the eager-PyTorch reference output for the same inputs (the golden).

2. **Run** the `.pte` through the runtime on the same inputs:
   - via the C++ runner: `./cmake-out/executor_runner --model_path /tmp/<name>.pte` (or the project's runner), or
   - via pybindings: load the program and execute the forward method in Python.

3. **Assert** every acceptance criterion from the charge:
   - **Parity:** runtime output vs eager reference within the declared tolerance (e.g. `max abs err < 1e-3`). A criterion that can't be observed = fail.
   - Any charge-specific bound: the expected backend/delegate was used, latency/memory budget held, output shapes/dtypes correct.

4. **Certify** (only on full pass):
   ```sh
   SHA=$(git rev-parse HEAD)
   TESTS=$(jq -r '.tests // false' .claude/state/green.json 2>/dev/null || echo false)
   jq -n --arg sha "$SHA" --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson tests "$TESTS" \
     '{sha:$sha, tests:$tests, uat:true, ts:$ts}' > .claude/state/green.json
   ```

5. **Record** evidence — the `.pte` path and the observed-vs-reference numbers — in the return.

## Rule
Never set `uat:true` for a run you did not actually observe passing. The sentinel authorizes a production merge. A parity check you skipped or loosened = fail.
