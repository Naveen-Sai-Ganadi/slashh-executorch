"""Guard the CI workflow config (.github/workflows/ci.yml).

The host test suite only protects the project if CI actually runs it. This test
keeps the workflow well-formed and pointed at the right command, so an edit that
silently breaks CI (bad YAML, dropped pytest step, wrong Python) fails locally
first. It asserts structure, not the exact wording. Host-only; no token.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _load() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def test_workflow_exists_and_parses() -> None:
    assert _WORKFLOW.is_file()
    assert isinstance(_load(), dict)


def test_workflow_runs_the_suite_on_push() -> None:
    wf = _load()
    # PyYAML parses the bare `on:` key as the boolean True; accept either form.
    triggers = wf.get("on", wf.get(True))
    assert "push" in triggers

    jobs = wf["jobs"]
    assert jobs, "no jobs defined"
    steps = next(iter(jobs.values()))["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    # installs the verified pins and runs pytest over tests/
    assert "model/requirements.txt" in run_cmds
    assert "pytest" in run_cmds and "tests/" in run_cmds
    # pins Python 3.11 (the verified interpreter)
    assert "3.11" in run_cmds
