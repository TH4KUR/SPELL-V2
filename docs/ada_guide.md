# Running things on Ada — quick guide

Short operational reference for actually getting scripts/sbatch jobs running
on the Ada cluster. This is NOT a substitute for `PROTOCOL.md` (§5 especially)
— that's the authoritative source; this is the "how do I not get stuck for an
hour" version, distilled from real incidents this project has already hit.

**Golden rule**: the user runs every `sbatch`/`srun`/`scancel` themselves.
An assistant's SSH access here is READ-ONLY diagnosis (checking `squeue`,
`sinfo`, log files) — write the commands, hand them over, let the user run
them and paste back output.

## 1. Getting on and around

- `ssh <user>@ada.iiit.ac.in` lands you on the **login node**. It runs NO
  usable Python (bare `python` is CentOS-7 2.7). It CAN see `/share1`.
- **Compute nodes** (where jobs actually run) have the real Python via the
  frozen env, but **cannot see `/share1` at all** (verified, not a mount
  option — genuinely no path there). This split matters constantly: to read
  something from `/share1` (e.g. an already-archived run) AND run python on
  it, you must first `rsync` it from `/share1` to somewhere under `$HOME`
  on the login node (plain file copy, no python needed), THEN process it
  from a compute node.

## 2. Environment activation — order matters, always

```bash
module purge
module load u22/python/3.12.4
source ~/envs/spell/bin/activate
python ...            # never `python3`, never bare `python` before this
```
Skipping the module load, or running `python` before `source activate`,
both fail in different unhelpful ways (this bit the project twice on the
same day early on). `template.sbatch` and `setup_env.sbatch` have the
correct order memorized — copy from those, not from memory.

## 3. The scheduling string is frozen — always use it verbatim

```
-p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066
```
CPU-only jobs add `--gres=gpu:0`; GPU jobs add `--gres=gpu:1` (or more).
Never hand-roll a different combination — this exact string is copy-paste
safe and guards against landing on a known-bad node or a drifted GPU.

**Gotcha — pinning a specific node**: if you ever add `--nodelist=gnodeXXX`,
that node must actually carry the `2080ti` SLURM feature tag, or combining
it with `--constraint=2080ti` creates an impossible request that queues
forever with no error. Check first: `sinfo -N -n gnodeXXX -o '%N %t %f'`.
Normally, just omit `--nodelist` entirely and let SLURM pick any free node.

## 4. Per-user limits on the `medium` QOS (checked directly, not assumed)

```
MaxSubmitPU = 8    # running + PENDING jobs combined, per user
MaxJobsPU   = 4    # of those, at most 4 may actually be RUNNING at once
```
Every SLURM **array task** counts as its own job against both. Before
submitting an array of size N, check `squeue -u $USER | wc -l` and keep
(N + existing) ≤ 8, or split into smaller batches submitted over time.

**`--time` gotcha**: omitting `--time` does NOT fall back to the QOS's
generous `MaxWall` (4 days) — it falls back to the **partition's**
`DefaultTime`, a separate and much shorter setting (currently 1 hour on
`u22`). Always pass an explicit `--time`, sized from real observed
throughput with margin — never omit it, never assume a QOS setting fills
the gap. Check either with `scontrol show partition u22 | grep DefaultTime`
/ `sacctmgr show qos medium format=Name,MaxWall,MaxSubmitPU,MaxJobsPU`.

## 5. Running a batch job

```bash
sbatch slurm/score_dnsmos.sbatch                       # standalone script
sbatch --array=0-7 --export=ALL,SPELL_RUN_PLAN=$HOME/spell/repo/slurm/run_plan_p3_selectors.tsv \
       slurm/template.sbatch                            # run-plan-driven array
```
`slurm/template.sbatch` is the generic driver for anything keyed off a
run-plan TSV (`scripts/gen_run_plan.py` writes these — never hand-edit one).
It reads `$SPELL_RUN_PLAN`, matches `$SLURM_ARRAY_TASK_ID` against column 1
of the TSV (never by physical line position), and dispatches to
`scripts/train_<track>.py` with the right manifest/seed/config.

Check status: `squeue -u $USER`. After it finishes: `sacct -j <jobid>
--format=JobID,State,ExitCode,Elapsed,Timelimit` tells you whether it
completed, timed out, or was cancelled, and how long it actually took.

## 6. Interactive sessions (debugging, one-off checks)

```bash
srun -p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066 \
     --gres=gpu:1 -N1 -n1 -t 30 --pty bash
```
`-t 30` = 30-minute wall limit; adjust for the task. Drop `--gres=gpu:1`
(use `--gres=gpu:0`) for CPU-only work. Never leave an interactive
allocation idle — release it with `exit` when done; `squeue --me` should
show zero stale rows before submitting something else.

## 7. Getting code changes onto Ada

Git-only, one direction of truth:
```
laptop: git push ada main
ada:    cd ~/spell/repo && git pull origin main
```
`ada` (an SSH remote) is the canonical repo; a GitHub mirror (`origin`) is
push-only convenience from the laptop side. Never `rsync` into a git
working tree, never hand-edit a tracked file directly on Ada — both cause
silent divergence that's hard to debug later.

## 8. Where the real detail lives

- `PROTOCOL.md` §5 — the full storage/environment/scheduling policy, with
  dated incident write-ups for every gotcha above (search for the exact
  symptom you're hitting; it's very likely already diagnosed there).
- `PROTOCOL.md` §10 — permanent known-issue list (hardware drift, node
  exclusions, single-writer completion-marker law, etc.).
- `KNOWN_BAD_NODES.md` — specific nodes with confirmed hardware problems.
