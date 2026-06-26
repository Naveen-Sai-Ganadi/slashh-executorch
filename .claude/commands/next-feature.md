---
description: Run exactly one feature end-to-end (design → TDD → implement → QA → UAT → ship).
---

Run a single `feature-pipeline` pass and stop. Steps:

1. Dispatch `product-strategist` to select the top backlog item (or use the item named in `$ARGUMENTS`) and write the charge.
2. Create a feature branch and run the full `feature-pipeline`: design → red tests → implement → QA → UAT (export a `.pte`, run it through the runtime, check parity) → `ship-it`.
3. Report the result: item, shipped?, merge SHA, follow-ups.

Do not start a second item. Respect the safety boundary.

$ARGUMENTS
