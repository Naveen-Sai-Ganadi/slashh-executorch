#!/bin/sh
# Shared helpers for Autonomy OS hooks.
# IMMUTABLE BY POLICY: the self-improve skill must never weaken the boundary
# this file (and git-guard.sh) enforces.

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
STATE_DIR="$REPO_ROOT/.claude/state"
SENTINEL="$STATE_DIR/green.json"

# Read the "command" field from the hook's stdin JSON payload.
hook_command() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '.tool_input.command // empty'
  else
    sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)".*/\1/p'
  fi
}

current_branch() { git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null; }
head_sha()       { git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null; }

# Prefer the project's virtualenv interpreter when present, else system python.
project_python() {
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    echo "$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    echo python
  fi
}

# A sentinel is "fresh" only when it certifies the current HEAD with both
# tests AND uat green. This is what authorizes a merge/push to main.
sentinel_is_fresh() {
  [ -f "$SENTINEL" ] || return 1
  command -v jq >/dev/null 2>&1 || return 1
  s_sha=$(jq -r '.sha // empty' "$SENTINEL" 2>/dev/null)
  s_tests=$(jq -r '.tests // false' "$SENTINEL" 2>/dev/null)
  s_uat=$(jq -r '.uat // false' "$SENTINEL" 2>/dev/null)
  [ "$s_sha" = "$(head_sha)" ] && [ "$s_tests" = "true" ] && [ "$s_uat" = "true" ]
}
