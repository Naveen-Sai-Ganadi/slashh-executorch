#!/bin/sh
# Stop: run the offline test suite. On success, record tests:true in the
# sentinel for the current HEAD (preserving any existing uat flag). On failure,
# clear the sentinel so nothing can ship. Always exits 0 (advisory to the loop).

DIR=$(dirname "$0"); . "$DIR/lib.sh"
cd "$REPO_ROOT" || exit 0
mkdir -p "$STATE_DIR"

PY=$(project_python)

# Skip when there is no pytest / no tests yet (e.g. a fresh scaffold).
"$PY" -c "import pytest" >/dev/null 2>&1 || exit 0
ls test tests */test */tests >/dev/null 2>&1 || \
  git ls-files '*_test.py' 'test_*.py' | grep -q . || exit 0

LOG=/tmp/etx-test-gate.log
# Quiet, offline run. -p no:cacheprovider keeps it side-effect free.
if "$PY" -m pytest -q -p no:cacheprovider >"$LOG" 2>&1; then
  prev_uat=false
  if [ -f "$SENTINEL" ] && command -v jq >/dev/null 2>&1; then
    p=$(jq -r '.uat // false' "$SENTINEL" 2>/dev/null); [ "$p" = "true" ] && prev_uat=true
  fi
  printf '{"sha":"%s","tests":true,"uat":%s,"ts":"%s"}\n' \
    "$(head_sha)" "$prev_uat" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$SENTINEL"
  echo "[test-gate] tests green; sentinel updated (uat=$prev_uat)" >&2
else
  rm -f "$SENTINEL"
  echo "[test-gate] tests FAILED — sentinel cleared; see $LOG" >&2
fi
exit 0
