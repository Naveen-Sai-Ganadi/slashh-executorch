---
description: Cross-role status report — backlog, recent ships, sentinel state, risks, budget.
---

Produce a concise standup across all roles. Gather and summarize:

1. **Backlog** — read `docs/autonomy/BACKLOG.md`: top unblocked items, anything `blocked` (with reason), what's `done`.
2. **Recent ships** — tail `.claude/state/run-log.jsonl`: last few iterations, shipped SHAs, failures.
3. **Ship readiness** — read `.claude/state/green.json`: is the current HEAD certified (tests/uat)?
4. **Risks & budget** — CFO note on token spend vs value shipped; any item spinning without progress.
5. **Next** — the single item the loop will pick next and why.

Keep it to a short, skimmable report.

$ARGUMENTS
