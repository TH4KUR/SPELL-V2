#!/bin/bash
# Push the staged Phase-0b dataset (~/spell/data) from this box to Ada.
#
# Wraps the tar-over-ssh stream with the checks that make it trustworthy:
#   preflight : source tree exists AND carries manifest.parquet (never ship
#               an unstaged tree); remote reachable; remote free-space sanity
#   transfer  : raw tar stream (payloads are already compressed), pipefail so
#               a dropped SSH fails loudly, pv progress bar when available,
#               resumable by simply rerunning (extraction overwrites cleanly)
#   postflight: remote inode count must match local; prints the exact
#               stage_to_ada.py --verify command that closes the gate
#
# Usage:
#   bash scripts/push_data_to_ada.sh [-n] [host] [remote_base]
#     -n            dry-run: run all checks, transfer nothing
#     host          default $SPELL_ADA_HOST (eashaan.thakur@ada.iiit.ac.in)
#     remote_base   default ~/spell (dataset lands at <remote_base>/data)

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "-n" ]] && { DRY_RUN=1; shift; }

HOST="${1:-${SPELL_ADA_HOST:-eashaan.thakur@ada.iiit.ac.in}}"
REMOTE_BASE="${2:-\$HOME/spell}"          # evaluated ON ada by the remote shell
LOCAL_DIR="$HOME/spell/data"
REMOTE_MIN_FREE_MB=6000                   # staged tree is ~4.5 GB

echo "== push_data_to_ada =="
echo "   host        : $HOST"
echo "   local  src  : $LOCAL_DIR"
echo "   remote dest : $HOST:$REMOTE_BASE/data"

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

# ---------- transfer ----------
echo "-- streaming tar -> $HOST (this moves ~${LOCAL_GB} MB / ~100k small files)..."
# NOTE: build the pipeline conditionally — an empty $PV would yield a syntax error.
if command -v pv >/dev/null 2>&1; then
    tar -C "$HOME/spell" -cf - data \
        | pv -s "${LOCAL_GB}M" \
        | ssh "${SSH_OPTS[@]}" "mkdir -p '$REMOTE_BASE' && tar -xpf - -C '$REMOTE_BASE'"
else
    tar -C "$HOME/spell" -cf - data \
        | ssh "${SSH_OPTS[@]}" "mkdir -p '$REMOTE_BASE' && tar -xpf - -C '$REMOTE_BASE'"
fi
echo "-- stream finished."

# ---------- postflight ----------
REMOTE_INODES=$(ssh "${SSH_OPTS[@]}" "find '$REMOTE_BASE/data' | wc -l")
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
