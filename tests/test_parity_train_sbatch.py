"""slurm/parity_train.sbatch follows the frozen cluster conventions (PROTOCOL §5.0, §5 items 8/8a/13,
§3.31; docs/ada_guide.md), selects the recipe by environment, supports resumption and never
unlocks the test split."""

from __future__ import annotations

import re
import subprocess

from config import PROJECT_ROOT

SBATCH = PROJECT_ROOT / "slurm" / "parity_train.sbatch"
LINES = SBATCH.read_text(encoding="utf-8").splitlines()
DIRECTIVES = [ln for ln in LINES if ln.startswith("#SBATCH")]
CODE = [ln for ln in LINES if ln.strip() and not ln.strip().startswith("#")]
TEXT = "\n".join(CODE)


def test_frozen_scheduling_string_is_present_verbatim():
    text = "\n".join(DIRECTIVES)
    for needle in ("-p u22", "-A research", "--qos=medium", "--constraint=2080ti",
                   "--exclude=gnode066", "--gres=gpu:1"):
        assert needle in text, needle


def test_directives_are_flags_only_with_no_inline_comments():
    assert all("#" not in ln[len("#SBATCH"):] for ln in DIRECTIVES)


def test_time_is_explicit_and_within_the_qos_wall_limit():
    times = [ln for ln in DIRECTIVES if "--time" in ln]
    assert len(times) == 1
    d, hms = re.fullmatch(r"#SBATCH --time=(?:(\d+)-)?(\d+:\d+:\d+)", times[0]).groups()
    hours = int(d or 0) * 24 + int(hms.split(":")[0])
    assert 24 <= hours <= 96                      # MaxWall is 4 days (§5 item 8a)
    assert "UNMEASURED ON ADA" in SBATCH.read_text()


def test_repo_root_comes_from_slurm_submit_dir_never_dollar_zero():
    assert 'REPO_ROOT="${SLURM_SUBMIT_DIR}"' in CODE
    assert not any('dirname "$0"' in ln for ln in LINES)
    assert 'cd "$REPO_ROOT"' in CODE


def test_environment_is_loaded_in_the_documented_order_before_any_python():
    idx = {name: next(i for i, ln in enumerate(CODE) if pat in ln) for name, pat in {
        "purge": "module purge", "load": "module load u22/python/3.12.4",
        "activate": "source ~/envs/spell/bin/activate"}.items()}
    assert idx["purge"] < idx["load"] < idx["activate"]
    first_python = next(i for i, ln in enumerate(CODE) if re.match(r"\s*python\s", ln))
    assert idx["activate"] < first_python


def test_gates_run_before_training_in_order():
    storage = next(i for i, ln in enumerate(CODE) if "scripts/check_storage.py --strict" in ln)
    guard = next(i for i, ln in enumerate(CODE) if "hardware_guard.enforce_gpu_policy()" in ln)
    train = next(i for i, ln in enumerate(CODE) if "scripts/parity_train.py" in ln)
    assert storage < guard < train


def test_data_root_defaults_to_the_canonical_constant_and_is_existence_gated():
    assert ': "${SPELL_DATA_ROOT:=$HOME/spell/data}"' in CODE
    assert any("export SPELL_DATA_ROOT" in ln for ln in CODE)
    assert any('[[ -d "$SPELL_DATA_ROOT" ]]' in ln for ln in CODE)


def test_the_script_never_sets_the_test_unlock_itself():
    for ln in LINES:
        assert "SPELL_FINAL_EVAL" not in ln.split("#", 1)[0], ln
    assert "--split" not in TEXT                          # the training job has no test path at all


def test_no_laptop_only_environment_leaks_in():
    text = SBATCH.read_text(encoding="utf-8")
    assert "pymax" not in text and "SPELL-V2" not in text and "SPELL_DEV_GPU" not in text


def test_recipe_is_chosen_by_environment_and_validated():
    assert 'RECIPE="${PARITY_RECIPE:?' in TEXT
    assert 'case "$RECIPE" in R-A|R-B) ;;' in TEXT
    call = " ".join(CODE[next(i for i, ln in enumerate(CODE) if "scripts/parity_train.py" in ln):][:7])
    assert '--recipe "$RECIPE"' in call and '--out-dir "$OUT"' in call and '--job-id "${SLURM_JOB_ID}"' in call
    assert '--tokens-dir "$TOKENS_DIR"' in call and '${EXTRA[@]+"${EXTRA[@]}"}' in call


def test_outputs_go_under_outputs_parity_train_recipe_runid():
    assert 'OUT="outputs/parity/train_${RECIPE}_${RUN_ID}"' in CODE
    assert 'RUN_ID="${PARITY_RUN_ID:-$SLURM_JOB_ID}"' in CODE


def test_resume_needs_the_first_job_id_and_an_existing_last_checkpoint_and_fresh_runs_never_overwrite():
    assert '[[ "${PARITY_RESUME:-0}" == "1" ]]' in TEXT
    assert 'fatal "PARITY_RESUME=1 needs PARITY_RUN_ID=<first job id>"' in TEXT
    assert '[[ -f "$OUT/last.pt" ]]' in TEXT and "EXTRA+=(--resume)" in TEXT
    assert '[[ ! -e "$OUT" ]] || fatal' in TEXT


def test_tokenizer_artifacts_are_required_and_the_failure_says_where_they_come_from():
    assert 'TOKENS_DIR="${PARITY_TOKENS_DIR:-parity/spm}"' in CODE
    assert '[[ -f "$TOKENS_DIR/tokenizer.json" ]]' in TEXT and "parity_train_spm.py" in TEXT


def test_splits_are_built_in_the_job_only_when_absent_like_the_eval_job():
    guard = TEXT.index("[[ ! -f parity/splits/provenance.json ]]")
    assert guard < TEXT.index("scripts/parity_build_splits.py") < TEXT.index("scripts/parity_train.py")


def test_fp32_determinism_environment_is_exported_before_python_runs():
    export = next(i for i, ln in enumerate(CODE) if ln.startswith("export CUBLAS_WORKSPACE_CONFIG"))
    first_python = next(i for i, ln in enumerate(CODE) if re.match(r"\s*python\s", ln))
    assert export < first_python


def test_optional_flags_forward_only_when_their_variable_is_set():
    for var, flag in (("PARITY_STOP_AFTER_EPOCHS", "--stop-after-epochs"), ("PARITY_BEAM_VAL_EVERY", "--beam-val-every"),
                      ("PARITY_BEAM_VAL_LIMIT", "--beam-val-limit"),
                      ("PARITY_MICRO_BATCH_SECONDS", "--micro-batch-seconds")):
        assert f'if [[ -n "${{{var}:-}}" ]]; then EXTRA+=({flag} "${var}"); fi' in TEXT, var
    assert 'if [[ "${PARITY_NONDETERMINISTIC:-0}" == "1" ]]; then EXTRA+=(--no-deterministic-kernels); fi' in TEXT


def test_bash_syntax_is_valid():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)


def test_usage_documents_timing_full_and_resume_commands():
    text = SBATCH.read_text(encoding="utf-8")
    assert "PARITY_STOP_AFTER_EPOCHS=1" in text
    assert "PARITY_RECIPE=R-A" in text and "PARITY_RECIPE=R-B" in text
    assert "PARITY_RESUME=1,PARITY_RUN_ID=" in text
    assert "mkdir -p logs" in text
