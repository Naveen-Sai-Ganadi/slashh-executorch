---
description: Run end-to-end UAT (export a model, run it through the runtime, check parity) and certify the UAT sentinel on pass.
---

Dispatch `uat-runner` and follow the `uat-example` skill against the current working change:

1. Export the model under test (from `$ARGUMENTS` or the active backlog item) to a `.pte` using the AOT stack.
2. Run it through the runtime (`executor_runner` / pybindings) on representative inputs and compare the output against the eager-PyTorch reference within tolerance.
3. On a full pass, set `uat:true` in `.claude/state/green.json` for the current HEAD. On fail, report what broke (observed vs reference numbers) and leave the sentinel untouched.

$ARGUMENTS
