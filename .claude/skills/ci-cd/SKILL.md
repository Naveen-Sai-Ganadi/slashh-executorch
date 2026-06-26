---
name: ci-cd
description: Use to create or maintain CI/CD pipelines — GitHub Actions for Python + C++ build/test, artifact/release pipelines, secrets policy, and deploy gating. Run by devops-engineer.
---

# CI/CD

Set up and keep green the pipelines that build, test, and ship slashh-executorch — one repo with an AOT (Python) world and a runtime (C++) world joined at the `.pte` contract.

## Where CI lives
- **`.github/workflows/ci.yml`** — the comprehensive gate: a Python job and a C++ job. Offline and deterministic (no model/weight downloads, pinned toolchains).

## Steps
1. **Dispatch `devops-engineer`** to author/update the workflows for the target(s).
2. **CI:**
   - Python job: set up Python, install the package, run `pytest -q`, optional `mypy`. Cache pip.
   - C++ job: set up CMake + a compiler, configure + build, run `ctest`. Cache the build dir where possible.
   - Matrix: Linux + macOS at minimum (the runtime targets many devices). Add more targets as backends land.
3. **CD / release** (`.github/workflows/release.yml`): on a `v*` tag, build the Python wheel and the
   runtime/`executor_runner` artifacts and attach them to a GitHub Release. Gate any signed/native
   build on its secret via `workflow_dispatch` or a tag job.
4. **Validate** YAML (`python -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))'` or `actionlint`).
5. **Publish** — workflows ship on the normal repo flow (sentinel-gated via `ship-it`/`release-manager`).

## Rules
- Never commit secrets; declare required ones (signing keys, registry tokens) and use Actions secrets / OIDC.
- Keep the gate offline and reproducible; releases honor the green sentinel + CI; never weaken the git-guard.

## Return
Return: workflow paths, what each gates, required secrets, and release-trigger instructions.
