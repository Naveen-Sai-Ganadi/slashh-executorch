---
description: Curate Claude Code's own state — record a durable fact, preference, decision, or convention in the right surface (memory, skill, agent, command, or settings).
---

Invoke the `config-curator` skill. Take whatever durable change is in play — a user preference or
correction, a project fact/decision/constraint, a new convention, or a stale/missing
skill/agent/command/setting — and route it to the correct surface:

- a *fact to remember* → a **memory** file (+ a `MEMORY.md` pointer),
- a *repeatable procedure* → a **skill**,
- a *role* → an **agent**,
- an *invocable entry point* → a **command**,
- *tool wiring / a gate* → **settings**.

Dedupe before writing (update, don't duplicate), keep secrets out of every tracked surface, never
weaken the safety boundary, log the change to `.claude/state/run-log.jsonl`, and commit tracked
surfaces with the co-author trailer (memory lives outside the repo — just write it).

If `$ARGUMENTS` names the thing to capture, curate that; otherwise scan the recent conversation for
what should be persisted and propose it.

$ARGUMENTS
