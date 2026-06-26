---
name: devrel
description: Use post-ship or on demand to run the developer-relations motion — changelog/release notes, README/docs refresh, examples index — then publish OWNED surfaces while leaving outbound as human-gated drafts.
---

# DevRel (developer-relations motion)

Keep docs in pace with shipping. Runs after a feature ships green, or on demand via `/devrel`.

## Steps

1. **Dispatch `devrel-lead`** with the latest ship(s) and the product vision. It produces:
   - a `docs/devrel/changelog.md` entry + a release note,
   - refreshed README / `docs/` where the ship changes the getting-started or export→run story,
   - an updated `examples/` index when new model paths land,
   - outbound drafts in `docs/devrel/outreach/` (human-gated).

2. **Publish OWNED surfaces only** (this step is run by `release-manager`/`ship-it`, which hold git —
   `devrel-lead` itself has no Bash):
   - Commit README + `docs/` + `examples/` index + `docs/devrel/` changes (co-author trailer).
   - Push to `origin main` — within the safety boundary (a fresh green sentinel is required like any push to `main`).
   - **Do NOT touch `docs/devrel/outreach/`** beyond writing the drafts — nothing there is ever sent.

3. **Log** the publish in `.claude/state/run-log.jsonl`.

## Send-tier boundary (must hold)
- Owned surfaces (README, docs, changelog, release notes, examples index) → auto-publish ✅.
- Outbound to real people / external communities (blog, forum, Discord, email) → drafts only, human posts 🔒.
- No agent has a send/post tool. If a need arises to actually message a person, STOP and hand the draft to the human.

## Return
Return: published owned-surface targets, queued outreach drafts, and confirmation nothing was sent to a person.
