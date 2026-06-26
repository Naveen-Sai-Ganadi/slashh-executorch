---
name: build-loop
description: Use to run the autonomous build loop — continuously groom the backlog, pick the top item, build it through the feature pipeline, log, repeat. The thing /loop runs.
---

# Build Loop

The master autonomous iteration. One pass = one shipped feature. Designed to run under `/loop` so it repeats until the backlog is empty.

## One iteration

1. **Re-establish context** — read the Autonomy Charter memory, `docs/autonomy/BACKLOG.md`, and `.claude/state/run-log.jsonl`.
2. **Groom** — invoke `backlog-grooming` to ensure the top items are well-formed and prioritized.
3. **Select** — dispatch `product-strategist` to pick the top unblocked item and write the charge (with a token budget ceiling).
4. **Build** — invoke `feature-pipeline` for that item. It ends with a ship to `main` (or a logged failure).
5. **Log** — append the result to `.claude/state/run-log.jsonl`.
6. **Budget check (CFO)** — if the iteration blew its budget without shipping, record why and shrink the next item.

## Stopping
- If the backlog has no unblocked must-have items left: dispatch `product-strategist` to declare the product **ready**, then switch to **DevRel mode** — run the `devrel` skill (`devrel-lead` as full developer-relations operator): keep the README/docs/changelog current and auto-published, the `examples/` index fresh, and queue outbound drafts for the human. Keep iterating DevRel/backlog items rather than stopping cold.
- If a `feature-pipeline` fails twice on the same item, mark it `blocked` with the reason and move to the next item (don't spin).

## Self-evolution
When the loop repeatedly hits the same gap (a recurring manual step, a missing recipe), invoke `self-improve` to add the skill/agent/command that closes it — without asking the human. Never weaken the safety boundary.

## Return
Per pass, return a one-line summary: item, shipped?, SHA, next item. Under `/loop`, continue to the next iteration.
