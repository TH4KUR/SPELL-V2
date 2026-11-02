"""slurm/parity_eval.sbatch follows the frozen cluster conventions (PROTOCOL §5.0,
§5 item 8/8a/13, §3.31; docs/ada_guide.md) and never unlocks the test split itself."""

from __future__ import annotations

import re

from config import PROJECT_ROOT

SBATCH = PROJECT_ROOT / "slurm" / "parity_eval.sbatch"
LINES = SBATCH.read_text(encoding="utf-8").splitlines()
DIRECTIVES = [ln for ln in LINES if ln.startswith("#SBATCH")]
CODE = [ln for ln in LINES if ln.strip() and not ln.strip().startswith("#")]


def test_frozen_scheduling_string_is_present_verbatim():
    text = "\n".join(DIRECTIVES)
    for needle in ("-p u22", "-A research", "--qos=medium", "--constraint=2080ti",
                   "--exclude=gnode066", "--gres=gpu:1"):
        assert needle in text, needle


def test_directives_are_flags_only_with_no_inline_comments():
    assert all("#" not in ln[len("#SBATCH"):] for ln in DIRECTIVES)


def test_time_is_explicit_and_sized():
    times = [ln for ln in DIRECTIVES if "--time" in ln]
    assert times == ["#SBATCH --time=04:00:00"]


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


def test_storage_gate_runs_strict_before_the_eval():
    storage = next(i for i, ln in enumerate(CODE) if "scripts/check_storage.py --strict" in ln)
    evalrun = next(i for i, ln in enumerate(CODE) if "scripts/parity_eval.py" in ln)
    assert storage < evalrun


def test_data_root_defaults_to_the_canonical_constant_and_is_existence_gated():
    assert ': "${SPELL_DATA_ROOT:=$HOME/spell/data}"' in CODE
    assert any("export SPELL_DATA_ROOT" in ln for ln in CODE)
    assert any('[[ -d "$SPELL_DATA_ROOT" ]]' in ln for ln in CODE)


def test_the_script_never_sets_the_test_unlock_itself():
    for ln in LINES:
        code = ln.split("#", 1)[0]
        assert "SPELL_FINAL_EVAL" not in code, ln


def test_no_laptop_only_environment_leaks_in():
    text = SBATCH.read_text(encoding="utf-8")
    assert "pymax" not in text and "SPELL-V2" not in text


def test_eval_gets_the_job_id_checkpoint_and_split_and_nothing_hardcoded():
    call = " ".join(CODE[next(i for i, ln in enumerate(CODE) if "scripts/parity_eval.py" in ln):][:4])
    assert '--checkpoint "$CKPT"' in call and '--split "$SPLIT"' in call
    assert '--job-id "${SLURM_JOB_ID}"' in call


def test_split_is_validated_and_valid_id_file_only_needed_when_splits_unbuilt():
    text = "\n".join(CODE)
    assert 'case "$SPLIT" in val|test) ;;' in text
    build = text.index("scripts/parity_build_splits.py")
    guard = text.index("[[ ! -f parity/splits/provenance.json ]]")
    assert guard < build


def test_optional_units_and_sha_pin_are_forwarded_only_when_set():
    text = "\n".join(CODE)
    assert 'EXTRA+=(--units "$PARITY_UNITS")' in text
    assert 'EXTRA+=(--expect-sha256 "$PARITY_EXPECT_SHA256")' in text
    assert '${EXTRA[@]+"${EXTRA[@]}"}' in text          # bash 4.2 (Ada) + set -u: empty array must not abort
    call = next(i for i, ln in enumerate(CODE) if "scripts/parity_eval.py" in ln)
    assert CODE[call - 1].startswith('echo "[parity_eval]') and text.index("EXTRA=()") < text.index("scripts/parity_eval.py")


def test_extra_args_expand_to_nothing_under_set_u_when_no_option_is_given(tmp_path):
    import subprocess
    script = tmp_path / "t.sh"
    script.write_text('set -euo pipefail\nEXTRA=()\nprintf "[%s]" ${EXTRA[@]+"${EXTRA[@]}"}\n'
                      'EXTRA+=(--units "a b")\nprintf "[%s]" ${EXTRA[@]+"${EXTRA[@]}"}\n')
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, check=True).stdout
    assert out == "[][--units][a b]"        # empty: printf sees no argument at all; set: two separate words
