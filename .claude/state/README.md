# Autonomy OS — shared state

This directory holds the Autonomy OS's runtime state. Do not edit by hand.

- `green.json` — the **ship sentinel**. Written by `test-gate.sh` (tests) and the
  `uat-example` skill (uat). Schema:
  ```json
  { "sha": "<HEAD sha>", "tests": true, "uat": true, "ts": "<iso8601>" }
  ```
  A merge/push to `main` is only allowed when this certifies the current HEAD
  with both `tests` and `uat` true. Here `uat` means an exported model was run
  end-to-end through the runtime and matched the eager-PyTorch reference within
  tolerance. Generated — gitignored.
- `run-log.jsonl` — append-only log of loop iterations (one JSON object per line).
  Generated — gitignored.
