# plan_reader.awk — THE run-plan row extractor (single source of truth).
#
#   awk -v t="$SLURM_ARRAY_TASK_ID" -f scripts/plan_reader.awk "$PLAN"
#
# Prints EXACTLY ONE line on stdout: the data row whose first TAB field equals
# t. Everything else is a FATAL on stderr with a nonzero exit:
#   2 = structural violation (no/incorrect header, wrong field count, empty
#       field, non-integer id, duplicate id)
#   3 = requested task id absent from the plan
# Position-based reads (line offsets from array id) are FORBIDDEN — the task id
# lives in the DATA (PROTOCOL §5 item 10). Run-plan files are written ONLY by
# scripts/gen_run_plan.py; this reader and that writer agree on the schema:
#
#   task | track | subset_manifest | seed | config_yaml | keep_local_traj
#
# Blank lines and lines starting with '#' are ignored anywhere in the file;
# the first non-comment row MUST be the literal header above.
function die(code, msg) {
    printf "FATAL[run_plan]: %s\n", msg > "/dev/stderr"
    failed = code
    exit code
}
BEGIN { FS = "\t" }
/^#/ || /^[[:space:]]*$/ { next }
!header_seen {
    if ($1 != "task")
        die(2, sprintf("line %d: first non-comment row must be the header 'task|%s|%s|%s|%s|%s', got '%s'",
                       FNR, "track", "subset_manifest", "seed", "config_yaml",
                       "keep_local_traj", $0))
    if (NF != 6)
        die(2, sprintf("line %d: header needs 6 tab-separated fields, got %d", FNR, NF))
    header_seen = 1
    next
}
{
    if (NF != 6)
        die(2, sprintf("line %d: need 6 tab-separated fields task|track|subset_manifest|seed|config_yaml|keep_local_traj, got %d ('%s')",
                       FNR, NF, $0))
    if ($1 !~ /^[0-9]+$/)
        die(2, sprintf("line %d: task id '%s' is not a non-negative integer", FNR, $1))
    if ($1 in rows)
        die(2, sprintf("duplicate task id %s (lines %d and %d)", $1, rows[$1], FNR))
    for (i = 1; i <= 6; i++)
        if ($i == "")
            die(2, sprintf("line %d: field %d is empty ('%s')", FNR, i, $0))
    rows[$1] = FNR
    order[++n] = $1
    if ($1 == t) {
        requested_seen = 1
        match_line = $0          # remember THE requested row, not merely the last
    }
}
END {
    # awk quirk: `exit` in a rule STILL runs END — honor the pending fatal here,
    # otherwise the "not found" check below would mask structural exit codes.
    if (failed) {
        exit failed
    }
    if (!header_seen)
        die(2, "empty plan: only comment/blank lines")
    if (!requested_seen)
        die(3, sprintf("task id %s not found; plan has %d row(s): ids %s",
                       t, n, join_ids()))
    print match_line
}
function join_ids() {
    s = ""
    for (j = 1; j <= n; j++) s = s (j > 1 ? "," : "") order[j]
    return s
}
