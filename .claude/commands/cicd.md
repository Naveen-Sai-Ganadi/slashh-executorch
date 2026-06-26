---
description: Set up or maintain CI/CD — Python + C++ build/test workflows, artifact/release pipelines, secrets, deploy gating.
---

Invoke the `ci-cd` skill (run by `devops-engineer`):

1. For the target(s) in `$ARGUMENTS` (or the whole repo if unspecified), author/update the GitHub
   Actions workflows: a Python job (install + `pytest` + optional `mypy`, offline) and a C++ job
   (CMake configure + build + `ctest`), on a Linux+macOS matrix where it matters.
2. Add/maintain CD: build the wheel + runtime/`executor_runner` artifacts on a `v*` tag and attach
   them to a GitHub Release; gate native/signed builds on the required secret.
3. Validate workflow YAML; keep the gate offline and deterministic (no model/weight downloads).
4. Report workflow paths, what each gates, required secrets, and how to cut a release.

Never commit secrets. Releases honor the green sentinel + CI.

$ARGUMENTS
