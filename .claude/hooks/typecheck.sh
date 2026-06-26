#!/bin/sh
# PostToolUse(Edit|Write|MultiEdit): fast, NON-BLOCKING Python typecheck.
# Advisory only — surfaces type errors so the agent self-corrects before
# shipping. Never blocks an edit (always exits 0).

DIR=$(dirname "$0"); . "$DIR/lib.sh"
cd "$REPO_ROOT" || exit 0

PY=$(project_python)

# Skip unless a static type checker is actually available in the environment.
if "$PY" -c "import mypy" >/dev/null 2>&1; then
  out=$("$PY" -m mypy --no-error-summary --ignore-missing-imports . 2>&1)
  rc=$?
elif command -v pyright >/dev/null 2>&1; then
  out=$(pyright 2>&1); rc=$?
else
  exit 0  # no typechecker installed — nothing to do
fi

if [ "$rc" -ne 0 ]; then
  echo "[typecheck] type issues — self-correct before shipping:" >&2
  echo "$out" | tail -20 >&2
fi
exit 0
