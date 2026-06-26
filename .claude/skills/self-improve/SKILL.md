---
name: self-improve
description: Use mid-run when the loop hits a missing capability — create or update your own skills, agents, commands, or backlog, log it, and continue WITHOUT asking the human. Never weakens the safety boundary.
---

# Self-Improve (self-evolution)

The Autonomy OS grows its own tooling. When a `feature-pipeline` or `build-loop` pass hits something it lacks — a recurring manual step, a missing recipe, an unclear role boundary — you fix the *system*, not just the feature, and keep going. You do **not** ask the human for permission for additive changes.

## What you MAY do autonomously
- **Add/append/update a skill** under `.claude/skills/<name>/SKILL.md`.
- **Add/update a subagent** under `.claude/agents/<name>.md`.
- **Add/update a slash command** under `.claude/commands/<name>.md`.
- **Add/reprioritize backlog items** in `docs/autonomy/BACKLOG.md`.
- **Add advisory hooks** that strengthen quality (extra lint/test/parity gates).

## Scope vs `config-curator`
This skill is the **loop-internal** path: when a `feature-pipeline`/`build-loop` pass hits a missing
*tooling* capability, close it and keep going. For the broader job of keeping Claude Code's own state
current — **persistent memory** (cross-session facts/preferences/decisions, which this skill does NOT
touch), plus deciding *which surface* a change belongs in — use [[config-curator]]. Both share the
same immutability guardrails below.

## What you MAY NEVER do (immutable by policy)
- Edit `.claude/hooks/git-guard.sh` or `.claude/hooks/lib.sh`'s `sentinel_is_fresh` logic.
- Weaken, bypass, or disable the safety boundary (the tests+UAT-green-before-main rule).
- Loosen a numerical-parity tolerance to make a UAT pass.
- Make `devrel-lead` auto-post outreach to real people.
- Remove the co-author trailer convention.

If a genuine need requires touching any of the above, STOP and surface it to the human with the reason. That is the one case where you ask.

## Procedure
1. Name the gap in one sentence.
2. Make the smallest additive change that closes it, following existing file conventions.
3. Record it in `.claude/state/run-log.jsonl`:
   ```json
   {"self_improve":"<gap>","added":["<path>"],"ts":"<iso>"}
   ```
4. Commit it (with the co-author trailer) and continue the interrupted pass.

## Return
Return what was added/changed and the resumed step.
