---
name: feature-pipeline
description: Use to take ONE feature from charge to shipped — design → TDD → implement → QA → UAT → ship. The core of the autonomous loop.
---

# Feature Pipeline

Take a single backlog item from idea to merged-on-`main`, dispatching the right role subagent at each stage. Run the stages in order; do not skip the gates.

## Preconditions
- You are on a fresh feature branch: `git checkout -b feature/<item-id>-<slug>`.
- You have a strategist charge (goal, acceptance criteria, lead role, budget).

## Stages

1. **Design** — dispatch `api-designer` with the charge. Get the design brief (API surface, error/edge behavior, docstrings/help copy). Skip only for purely internal changes with no public surface.
2. **Red tests** — dispatch `qa-engineer` to write failing tests from the acceptance criteria (`pytest` and/or `ctest`/gtest), including a numerical-parity check where relevant. Verify they fail for the right reason.
3. **Implement** — dispatch `export-engineer` (AOT) and/or `runtime-engineer` (runtime/kernels) to make the tests pass with the smallest change. Honor the three invariants (portability fail-safe; AOT/runtime separation; numerical parity).
4. **QA** — dispatch `qa-engineer` to run the full affected suite, confirm green, add regression tests for anything found.
5. **UAT** — dispatch `uat-runner` (see `uat-example`) to export a model, run it through the runtime, and certify `uat:true` in the sentinel on a parity pass. If it fails, loop back to stage 3.
6. **Ship** — invoke `ship-it`. It verifies the green sentinel and auto-merges to `main`.
7. **DevRel** — once the product is past v1 (product-ready declared), invoke the `devrel` skill so docs keep pace: changelog/release note, README/docs refresh, `examples/` index. Owned surfaces auto-publish; outreach stays human-gated. Skip for pre-v1 internal work.

## On a missing capability
If any stage needs tooling that doesn't exist yet (a new skill, agent, command), invoke `self-improve` to create it, then continue — without asking the human.

## After shipping
Append a result line to `.claude/state/run-log.jsonl`:
```json
{"item":"<id>","branch":"feature/<id>-<slug>","shipped":true,"sha":"<sha>","ts":"<iso>"}
```
Mark the item `done` in `docs/autonomy/BACKLOG.md`.

## Return
Return: item id, shipped (true/false), the merge SHA, and any follow-up items discovered.
