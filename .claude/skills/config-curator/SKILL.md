---
name: config-curator
description: Use WHENEVER something durable changes that Claude Code should persist across sessions — a user preference or correction, a project fact/decision/constraint, a new convention, or a stale/missing skill, agent, command, or setting. Routes the change to the RIGHT surface (memory file, skill, agent, command, or settings). Proactive — invoke without being asked.
---

# Config Curator (keep Claude Code's own state current)

The Autonomy OS only stays sharp if its *configuration* keeps pace with reality. This skill is the
operator that maintains every Claude-Code-owned surface — **persistent memory**, **skills**,
**agents**, **commands**, and **settings** — so what Claude knows and how it behaves never drifts
from how the project actually works. Fire it the moment you notice a durable change, not just when
asked. Curating the system *is* part of the job.

## The surfaces, and what belongs in each

| Surface | Location | Holds | Lifetime |
| --- | --- | --- | --- |
| **Memory** | `~/.claude/projects/-Users-chinnu-slashh-executorch-slashh-executorch/memory/` | Durable *facts* — who the user is, guidance/corrections, project decisions/constraints, reference pointers | Cross-session, until wrong |
| **Skill** | `.claude/skills/<name>/SKILL.md` | A reusable *procedure/recipe* | Until the procedure changes |
| **Agent** | `.claude/agents/<name>.md` | A *role/persona* subagent (with `tools:`) | Until the role changes |
| **Command** | `.claude/commands/<name>.md` | A *slash-command entry point* (body ends with `$ARGUMENTS`) | Until the entry point changes |
| **Settings** | `.claude/settings.json` (shared) / `settings.local.json` (local, gitignored) | Hook wiring, permissions | Until wiring changes |

## Routing — pick the right surface (ask in this order)

1. **Is it a fact to *remember*** (a preference, a decision, a constraint, a pointer)? → **Memory**. Not a procedure, not behavior — knowledge.
2. **Is it a repeatable *procedure*** I'd want to run the same way again? → **Skill**.
3. **Is it a *role*** with its own responsibilities and tool set? → **Agent**.
4. **Is it an *invocable entry point*** a human types as `/x`? → **Command** (usually a thin wrapper over a skill).
5. **Is it *tool wiring / a gate / a permission*?** → **Settings** (shared) or a hook (advisory only — see guardrails).

When two fit, prefer the most specific and the least duplicative. A fact that drives a procedure goes to memory *and* the procedure references it.

## When to fire (triggers — don't wait to be asked)

- The user states a **preference or correction** ("always use X", "don't do Y", "I prefer Z") → write a `feedback` memory.
- A **project decision or constraint** is made or discovered (a target device, a version pin, a security rule, a deadline) → write a `project` memory.
- You learn **who the user is** / their role/expertise → write/update the `user` memory.
- A **convention** is adopted (naming, commit trailer, file layout) → memory if it's a fact, or fold it into the relevant skill/agent.
- A **skill/agent/command/setting is stale or missing** — it names a file/flag/command that changed, or a recurring manual step has no recipe → update or create it.
- You keep **explaining the same thing** across turns → that's a memory or a skill waiting to be written.

## Procedures by surface

### Memory (the surface `self-improve` does NOT cover)
One fact per file. Frontmatter + body:
```markdown
---
name: <short-kebab-slug>
description: <one-line summary — used to decide relevance on recall>
metadata:
  type: user | feedback | project | reference
---

<the fact. For feedback/project, follow with:>
**Why:** <the reason it matters>
**How to apply:** <what to do differently>

<link related memories with [[their-slug]]>
```
Then add ONE pointer line to `MEMORY.md` (create it if absent): `- [Title](file.md) — hook`.
- **Dedupe first:** if a file already covers it, *update that file* — never create a near-duplicate. Delete a memory that turns out wrong.
- **Don't store what's already recorded** elsewhere — code structure, git history, or anything in CLAUDE.md/the repo. If asked to "remember" such a thing, capture only what was *non-obvious* about it.
- Memory is **point-in-time**: if a memory names a file/flag/function, verify it still exists before acting on it.

### Skill / Agent / Command
Match the existing files exactly as templates — same frontmatter keys, tone, and a `## Return` contract:
- **Skill:** `.claude/skills/<name>/SKILL.md`, frontmatter `name` + `description` (the description is the trigger — make it say *when* to use it).
- **Agent:** `.claude/agents/<name>.md`, frontmatter `name`, `description`, `tools:` (least privilege — only the tools the role needs).
- **Command:** `.claude/commands/<name>.md`, frontmatter `description`, body ends with `$ARGUMENTS`. Keep it a thin wrapper that invokes a skill.
- Keep cross-references consistent (e.g. a new skill referenced from `feature-pipeline`/`build-loop`; a new example added to the `examples/` index).

### Settings
- Shared, version-controlled config → `.claude/settings.json`. Validate JSON before saving.
- Machine-local permissions/allowlists → `settings.local.json` (gitignored) — never put another machine's or another project's local rules here.

## Guardrails (immutable — never weakened by this skill)
- **Never** edit `.claude/hooks/git-guard.sh` or `lib.sh`'s `sentinel_is_fresh` logic, and never weaken the tests+UAT-green-before-`main` safety boundary. (Same immutability list as `self-improve` — if a genuine need touches these, STOP and surface it to the human.)
- **No secrets** in any committed/tracked surface or in memory — API tokens, keys, credentials live only in local config (e.g. `~/.qai_hub/client.ini`) or named env vars. Memory may record *that* a secret exists and *where*, never its value.
- **Smallest change that closes the gap**, following existing conventions. Don't refactor the system to make one edit.

## Logging & committing
- Record config/skill/agent/command/settings changes in `.claude/state/run-log.jsonl`:
  ```json
  {"curate":"<what changed>","surface":"memory|skill|agent|command|settings","path":["<path>"],"ts":"<iso>"}
  ```
- Commit *tracked* surfaces (skills/agents/commands/settings) with the co-author trailer
  `Co-authored-by: Naveen-Sai-Ganadi <naveenganadi@gmail.com>`. **Memory files and `MEMORY.md` live outside the repo** (under `~/.claude/...`) — they are not committed; just write them.

## Relationship to `self-improve`
- `self-improve` = mid-loop self-evolution: when the **build loop** hits a missing *tooling* capability, add the skill/agent/command and continue without asking. Scoped to the loop.
- `config-curator` = the broader curator, **including memory** and proactive maintenance outside the loop. It owns the routing decision (which surface) and the cross-session memory the loop never touches.
- They share the same immutability guardrails. For a loop-internal tooling gap, `self-improve` is the specialized path; for "what should Claude remember / which surface does this belong in," use this skill.

## Return
Return: what changed, which surface it landed in, the path(s) written, the `MEMORY.md`/index/cross-reference updates made, and (for tracked surfaces) whether it was committed.
