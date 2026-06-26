#!/bin/sh
# PreToolUse(Bash) guard: enforces the auto-merge-to-main safety boundary.
#
# IMMUTABLE BY POLICY (boundary may never be WEAKENED). This precise matcher was
# authorized by the user to remove false positives — it still gates every real
# force-push and every ungated push to a publish remote's `main`, but ignores
# read-only commands that merely contain the substrings "push"/"-f"/"main".
#
# Boundary: a push that lands on `main` is blocked unless a FRESH green sentinel
# (tests AND uat green for the current HEAD) exists. Force-pushes are always
# blocked. Non-main branch pushes are allowed.

DIR=$(dirname "$0"); . "$DIR/lib.sh"
CMD=$(hook_command)
[ -n "$CMD" ] || exit 0

# Co-author advisory on commits (non-blocking; commits are not pushes).
case "$CMD" in
  *"git commit"*)
    echo "$CMD" | grep -q "Co-authored-by: Naveen-Sai-Ganadi" || \
      echo "[git-guard] reminder: add 'Co-authored-by: Naveen-Sai-Ganadi <naveenganadi@gmail.com>' trailer" >&2 ;;
esac

# Beyond here we only consider REAL git pushes. Require `git` and `push` in the
# same shell segment (no pipe/redirect between them), so e.g.
# `git remote -v | grep '(push)'` is NOT treated as a push.
echo "$CMD" | grep -Eq 'git[^|;&]*push' || exit 0

is_blocked=0; reason=""

# Force-push: explicit --force / --force-with-lease / standalone -f / +refspec
# appearing after the push token (within the same segment).
if echo "$CMD" | grep -Eq 'push[^|;&]*(--force|--force-with-lease)' \
   || echo "$CMD" | grep -Eq 'push[^|;&]* -f( |$)' \
   || echo "$CMD" | grep -Eq 'push[^|;&]* \+'; then
  is_blocked=1; reason="force-push is not allowed"
fi

# Lands on main: an explicit `main` ref in the push, or a bare push while HEAD
# is main (no explicit other-branch refspec).
lands_on_main=0
echo "$CMD" | grep -Eq 'push[^|;&]*\bmain\b' && lands_on_main=1
if [ "$(current_branch)" = "main" ]; then
  echo "$CMD" | grep -Eq 'push[^|;&]*(main|/|:|-origin)' || lands_on_main=1
fi

if [ "$lands_on_main" = "1" ] && ! sentinel_is_fresh; then
  is_blocked=1
  reason="blocked push to main: no fresh green sentinel (tests+UAT must pass for current HEAD)"
fi

if [ "$is_blocked" = "1" ]; then
  echo "[git-guard] $reason" >&2
  exit 2
fi
exit 0
