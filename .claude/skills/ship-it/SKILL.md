---
name: ship-it
description: Use to release a green feature branch to main within the safety boundary — verify the sentinel, auto-merge, confirm CI.
---

# Ship It (gated release)

Auto-merge a feature to `main` — but only when it is genuinely green. The `git-guard` hook is the backstop; this skill is the disciplined path.

## Preconditions (verify, do not assume)
1. On a feature branch (not `main`), working tree committed.
2. `.claude/state/green.json` certifies the **current HEAD** with `tests:true` AND `uat:true`:
   ```sh
   jq -e --arg sha "$(git rev-parse HEAD)" \
     'select(.sha==$sha and .tests==true and .uat==true)' .claude/state/green.json
   ```
   If this fails, STOP — re-run tests (`test-gate`) and/or UAT (`uat-example`) first. Never hand-edit the sentinel to pass.

## Merge procedure
Prefer a **fast-forward** merge so `main` points at the *exact* commit that was
tested and UAT'd — this keeps the sentinel valid (HEAD == certified sha) and is
what the `git-guard` freshness check requires.
```sh
git checkout main
git pull --ff-only origin main
git merge --ff-only feature/<id>-<slug>   # main now == the certified HEAD
git push origin main                       # git-guard allows: sentinel matches HEAD
```
If `main` has diverged and fast-forward is impossible, rebase the feature branch
onto `main`, then re-run the test-gate and `uat-example` to re-certify the new
HEAD before pushing. Never push a merge commit the sentinel does not certify.

## After merge
- Watch CI: `gh run watch` (or `gh run list --branch main --limit 1`). If CI goes red, revert the merge commit and report — do not leave `main` broken.
- For a release, tag `v<x.y.z>` on the certified sha and push the tag (CD builds the wheel + runtime artifacts).
- Append the ship to `.claude/state/run-log.jsonl` and mark the backlog item `done`.

## Return
Return: merged (true/false), the merge SHA, CI status (green/red), any release tag, and any rollback taken.
