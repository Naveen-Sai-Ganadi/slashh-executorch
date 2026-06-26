---
description: Run the developer-relations motion — changelog, README/docs, example index, outreach drafts; auto-publish owned surfaces, human-gate outbound.
---

Invoke the `devrel` skill:

1. Dispatch `devrel-lead` to produce changelog/release notes (`docs/devrel/`), refresh the README
   and `docs/` if the latest ship changes the story, keep the `examples/` index current, and queue
   outbound drafts in `docs/devrel/outreach/` (human-gated).
2. Publish OWNED surfaces only (commit README/`docs/`/`examples/` index + push to `origin main`),
   within the safety boundary (fresh green sentinel required). Leave `docs/devrel/outreach/` unsent.
3. Report published targets, the deploy, and the queued drafts.

Never post outbound to a real person or external community — drafts only.

$ARGUMENTS
