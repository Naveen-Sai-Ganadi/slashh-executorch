---
name: backlog-grooming
description: Use to maintain and prioritize docs/autonomy/BACKLOG.md — well-formed items, dependencies, and status — optionally mirrored to Linear.
---

# Backlog Grooming

Keep `docs/autonomy/BACKLOG.md` healthy so the loop always has a clear next item.

## Item schema
Each item is a `##` heading with this block:
```markdown
## [id] Title
- **status:** todo | in-progress | blocked | done
- **priority:** P0 | P1 | P2
- **role:** lead subagent (e.g. export-engineer, runtime-engineer)
- **depends-on:** [ids] or none
- **acceptance:**
  - criterion 1 (observable in UAT — e.g. "example X exports and runs with <1e-3 max abs error vs eager")
  - criterion 2
```

## Grooming actions
- Split any item too big for one `feature-pipeline` pass (one shippable deliverable each).
- Make acceptance criteria **observable** — a `uat-runner` must be able to check each one by exporting and running a model.
- Resolve priority ties with value ÷ effort; respect `depends-on`.
- Promote discovered follow-ups (from the run log) into new items.
- Keep `done` items at the bottom for history.

## Optional Linear mirror
If a Linear team is configured, mirror P0/P1 items to Linear issues via the Linear MCP (`mcp__plugin_linear_linear__*`). BACKLOG.md stays the source of truth.

## Return
Return the ordered list of unblocked items with ids and priorities.
