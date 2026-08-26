#!/bin/bash
# Push the staged Phase-0b dataset (~/spell/data) from this box to Ada.
#
# Wraps the rsync-over-ssh transfer with the checks that make it trustworthy:
#   preflight : source tree exists AND carries manifest.parquet (never ship
#               an unstaged tree); remote reachable; remote free-space sanity
#   transfer  : INCREMENTAL rsync — safe to interrupt (Ctrl-C, dropped SSH);
#               rerunning resumes, sending only missing/stale files
#   postflight: remote inode count must match local; prints the exact
#               stage_to_ada.py --verify command that closes the gate
#
# Tip: for long links run it under tmux/nohup so closing YOUR terminal doesn't
# interrupt it:  nohup bash scripts/push_data_to_ada.sh > push.log 2>&1 &
#
# Usage:
#   bash scripts/push_data_to_ada.sh [-n] [host] [remote_base]
#     -n            dry-run: run all checks, transfer nothing
#     host          default $SPELL_ADA_HOST (eashaan.thakur@ada.iiit.ac.in)
#     remote_base   default <remote-$HOME>/spell (dataset -> <remote_base>/data)

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "-n" ]] && { DRY_RUN=1; shift; }

HOST="${1:-${SPELL_ADA_HOST:-eashaan.thakur@ada.iiit.ac.in}}"
LOCAL_DIR="$HOME/spell/data"
REMOTE_MIN_FREE_MB=6000                   # staged tree is ~4.5 GB
# Remote base defaults to <remote-$HOME>/spell; resolved concretely after the
# first successful ssh so no tilde/$HOME quoting survives into rsync/df/find.

echo "== push_data_to_ada =="
echo "   host        : $HOST"
echo "   local  src  : $LOCAL_DIR"

# ---------- preflight ----------
[[ -d "$LOCAL_DIR" ]] || { echo "FATAL: no staged tree at $LOCAL_DIR (run scripts/stage_to_ada.py first)" >&2; exit 1; }
[[ -f "$LOCAL_DIR/manifest.parquet" ]] || {
    echo "FATAL: $LOCAL_DIR/manifest.parquet missing — refusing to ship an unmanifested tree." >&2
    echo "       Run: python scripts/stage_to_ada.py --dest $LOCAL_DIR" >&2
    exit 1;
}

LOCAL_INODES=$(find "$LOCAL_DIR" | wc -l)
LOCAL_GB=$(du -sm "$LOCAL_DIR" | cut -f1)
echo "   local tree  : $LOCAL_INODES inodes, ${LOCAL_GB} MB"

echo "-- remote reachability..."
SSH_OPTS=(-o ConnectTimeout=15)
if ! ssh "${SSH_OPTS[@]}" "$HOST" 'true'; then
    echo "FATAL: cannot reach $HOST (check VPN/keys; password auth needs an interactive terminal)" >&2
    exit 1
fi
if [[ -z "${REMOTE_BASE:-}" ]]; then
    REMOTE_HOME=$(ssh "${SSH_OPTS[@]}" "$HOST" 'cd && pwd')
    REMOTE_BASE="$REMOTE_HOME/spell"
fi
echo "   remote base : $HOST:$REMOTE_BASE"
FREE_MB=$(ssh "${SSH_OPTS[@]}" "$HOST" "df -Pm '$REMOTE_BASE' 2>/dev/null | tail -1 | awk '{print \$4}'")
if [[ -n "$FREE_MB" && "$FREE_MB" -lt "$REMOTE_MIN_FREE_MB" ]]; then
    echo "FATAL: remote has only ${FREE_MB} MB free under $REMOTE_BASE (< $REMOTE_MIN_FREE_MB MB)" >&2
    exit 1
fi
echo "   remote free : ${FREE_MB:-?} MB (ok)"

if (( DRY_RUN == 1 )); then
    echo "-- dry-run: preflight passed, transfer skipped."
    exit 0
fi

# ---------- transfer (INCREMENTAL: safe to interrupt, rerun resumes) ----------
echo "-- rsync -> $HOST:$REMOTE_BASE/data (~${LOCAL_GB} MB / ~100k small files)..."
rsync -a --partial --info=progress2 \
      -e "ssh ${SSH_OPTS[*]}" \
      "$LOCAL_DIR/" "$HOST:$REMOTE_BASE/data/"
echo "-- transfer finished."

# ---------- postflight ----------
REMOTE_INODES=$(ssh "${SSH_OPTS[@]}" "$HOST" "find '$REMOTE_BASE/data' | wc -l")
if [[ "$REMOTE_INODES" != "$LOCAL_INODES" ]]; then
    echo "WARN: inode mismatch local=$LOCAL_INODES remote=$REMOTE_INODES — rerun this" >&2
    echo "      script (idempotent), then investigate before trusting the tree." >&2
else
    echo "   inode match : $REMOTE_INODES (local == remote)"
fi

cat <<EOF

NEXT (closes the Phase-0b gate — run INSIDE the Ada repo clone, venv active):
  cd ~/spell/repo
  python scripts/stage_to_ada.py --dest ~/spell/data --verify
  # must print: missing=0 extra=0 hash_mismatch=0
EOF
