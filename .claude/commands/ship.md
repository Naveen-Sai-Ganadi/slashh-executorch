---
description: Release the current green feature branch to main within the safety boundary.
---

Invoke the `ship-it` skill for the current branch:

1. Verify `.claude/state/green.json` certifies the current HEAD with `tests:true` AND `uat:true`. If not, STOP and report what's missing (run `/uat` or fix tests first).
2. Fast-forward the feature branch into `main` and `git push origin main` (the `git-guard` hook permits this only with a fresh sentinel).
3. Watch CI; if red, revert and report. For a release, tag `v<x.y.z>` on the certified sha.

Never hand-edit the sentinel to force a ship.

$ARGUMENTS
