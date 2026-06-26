---
description: Start the autonomous build loop — groom, pick, build, test, UAT (export+run), ship, repeat.
---

Run the Autonomy OS continuously. Invoke the `build-loop` skill and keep iterating: each pass picks the top backlog item, runs it through the `feature-pipeline`, and auto-merges to `main` once tests **and** UAT (an exported model runs end-to-end with parity) are green.

To run hands-off, wrap this in the `/loop` skill so it self-paces across iterations. Stop when the backlog's must-have items are all shipped, then declare the product ready and switch to `devrel-lead` (developer-relations) mode.

Honor the safety boundary at all times: no merge to `main` without a fresh green sentinel. Use `self-improve` to close tooling gaps mid-run without asking.

$ARGUMENTS
