#!/usr/bin/env bash

# Run stage 3 from run_stage.sh: initialize the segmentation model with the
# stage-2 pruned WavLM and the non-WavLM modules from a fine-tuned segmentation
# checkpoint, then train it with separate WavLM and non-WavLM optimizers.
# Stages 1, 2, and 4 are omitted.

set -euo pipefail
ulimit -n 2048

recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$recipe_root/../.." && pwd)"
venv_root="$project_root/.venv"
source_train_conf="$recipe_root/conf/dual_opt_common_all.toml"

stage1_conf_name="s80_base_all"
stage2_conf_name="${stage1_conf_name}_further_distill"
stage2_dir="$recipe_root/exp/$stage2_conf_name"
stage2_ckpt="${PRUNED_CKPT:-$stage2_dir/avg_student_ckpt/pytorch_model.bin}"

stage3_conf_name="${stage2_conf_name}_diar_training"
exp_root="$recipe_root/exp"
stage3_dir="$exp_root/$stage3_conf_name"

finetune_ckpt="${FINETUNE_CKPT:-$project_root/recipes/diar_ssl/exp_step_1_2_finetune_wavlm/wavlm_updated_conformer_all/checkpoints/average_32_33_35_36_37/pytorch_model.bin}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-1134}"
resume="${RESUME:-0}"
dry_run="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="$gpu_list"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$source_train_conf" ]]; then
    echo "Missing stage-3 configuration: $source_train_conf" >&2
    exit 1
fi
if [[ ! -x "$venv_root/bin/python" || ! -x "$venv_root/bin/accelerate" ]]; then
    echo "Missing training executables under $venv_root/bin" >&2
    exit 1
fi

if [[ ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer, got: $num_processes" >&2
    exit 1
fi
IFS=',' read -r -a visible_gpus <<< "$gpu_list"
if (( num_processes > ${#visible_gpus[@]} )); then
    echo "NUM_PROCESSES=$num_processes exceeds visible GPUs in CUDA_VISIBLE_DEVICES=$gpu_list" >&2
    exit 1
fi
if [[ ! "$main_process_port" =~ ^[0-9]+$ ]] \
    || (( main_process_port < 1 || main_process_port > 65535 )); then
    echo "MAIN_PROCESS_PORT must be an integer from 1 through 65535, got: $main_process_port" >&2
    exit 1
fi
if [[ "$resume" != "0" && "$resume" != "1" ]]; then
    echo "RESUME must be 0 or 1, got: $resume" >&2
    exit 1
fi
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1, got: $dry_run" >&2
    exit 1
fi

if [[ "$dry_run" == "1" ]]; then
    dry_run_config="$recipe_root/<temporary-dir>/$stage3_conf_name.toml"
    dry_run_args=(
        launch
        --num_processes "$num_processes"
        --main_process_port "$main_process_port"
        run_dual_opt_continued_training.py
        -C "$dry_run_config"
        -M train
        -FT
        --ckpt_path "$finetune_ckpt"
        --pruned_ckpt_path "$stage2_ckpt"
    )
    if [[ "$resume" == "1" ]]; then
        dry_run_args+=(-R)
    fi

    echo "Dry run only; no directories or configuration files were created."
    echo "Stage-2 WavLM prerequisite: $stage2_ckpt"
    echo "Non-WavLM segmentation initializer: $finetune_ckpt"
    echo "Stage-3 experiment directory: $stage3_dir"
    printf 'Dry run stage 3 command:'
    printf ' %q' "$venv_root/bin/accelerate" "${dry_run_args[@]}"
    printf '\n'
    exit 0
fi

active_pipeline_launchers="$(
    pgrep -af '[r]un_stage(1|2)_only\.sh|[r]un_stage\.sh' 2>/dev/null \
        || true
)"
if [[ -n "$active_pipeline_launchers" ]]; then
    echo "A stage-1 or stage-2 pipeline launcher is still running." >&2
    echo "Refusing to start stage 3 until the earlier-stage launcher has exited." >&2
    exit 1
fi

active_pruning_processes="$(
    pgrep -af '[r]un_distill_prune.py|[a]pply_pruning.py' 2>/dev/null \
        || true
)"
if grep -Fq -- "$recipe_root" <<< "$active_pruning_processes"; then
    echo "A stage-1 or stage-2 pruning process is still running under $recipe_root." >&2
    echo "Refusing to start stage 3 until the earlier-stage process has exited." >&2
    exit 1
fi

active_stage3_processes="$(
    pgrep -af '[r]un_dual_opt_continued_training.py' 2>/dev/null \
        || true
)"
if grep -Fq -- "$stage3_conf_name.toml" <<< "$active_stage3_processes"; then
    echo "A stage-3 process is already running for $stage3_conf_name." >&2
    echo "Refusing to launch a duplicate stage-3 job." >&2
    exit 1
fi

if [[ ! -s "$stage2_ckpt" ]]; then
    echo "Stage 3 requires the completed stage-2 WavLM checkpoint:" >&2
    echo "  $stage2_ckpt" >&2
    echo "Wait for run_stage2_only.sh to exit successfully before starting stage 3." >&2
    exit 1
fi
if [[ ! -s "$finetune_ckpt" ]]; then
    echo "Missing fine-tuned segmentation checkpoint: $finetune_ckpt" >&2
    exit 1
fi

if [[ "$resume" == "0" && -e "$stage3_dir" ]]; then
    echo "Experiment directory already exists: $stage3_dir" >&2
    echo "Set RESUME=1 to continue it, or move it aside before starting a fresh run." >&2
    exit 1
fi
if [[ "$resume" == "1" ]]; then
    shopt -s nullglob
    resume_model_files=("$stage3_dir"/checkpoints/epoch_[0-9][0-9][0-9][0-9]/pytorch_model.bin)
    shopt -u nullglob
    if [[ ! -d "$stage3_dir/tb_log" || ${#resume_model_files[@]} -eq 0 \
        || ! -f "${resume_model_files[0]:-}" ]]; then
        echo "RESUME=1 requires tb_log and at least one epoch checkpoint under $stage3_dir" >&2
        exit 1
    fi
fi

cd "$recipe_root"

"$venv_root/bin/python" - \
    "$source_train_conf" \
    "$finetune_ckpt" \
    "$stage2_ckpt" \
    "$num_processes" \
    "$exp_root" \
    "$stage3_dir" <<'PY'
from collections.abc import Mapping
from pathlib import Path
import sys

import toml
import torch

(
    config_value,
    finetune_checkpoint_value,
    stage2_checkpoint_value,
    num_processes_value,
    expected_save_dir_value,
    expected_experiment_dir_value,
) = sys.argv[1:]

config_path = Path(config_value).resolve()
finetune_checkpoint = Path(finetune_checkpoint_value).resolve()
stage2_checkpoint = Path(stage2_checkpoint_value).resolve()
num_processes = int(num_processes_value)
expected_save_dir = Path(expected_save_dir_value).resolve()
expected_experiment_dir = Path(expected_experiment_dir_value).resolve()
recipe_root = config_path.parent.parent
config = toml.load(config_path)

if config["trainer"]["path"] != "trainer_dual_opt.Trainer":
    raise RuntimeError(
        "This runner expects trainer_dual_opt.Trainer, got "
        f"{config['trainer']['path']!r}"
    )
if config["model"]["path"] != "diarizen.models.eend.model_wavlm_conformer.Model":
    raise RuntimeError(
        "This runner expects model_wavlm_conformer.Model, got "
        f"{config['model']['path']!r}"
    )
if config["trainer"]["args"].get("freeze_wavlm", False):
    raise RuntimeError("Stage-3 dual-optimizer training requires freeze_wavlm=false")
if config.get("finetune", {}).get("finetune", False):
    raise RuntimeError(
        "The source TOML must have finetune=false; the launcher supplies the "
        "selective initialization checkpoint explicitly"
    )

save_dir = Path(config["meta"]["save_dir"]).expanduser()
if not save_dir.is_absolute():
    save_dir = (recipe_root / save_dir).resolve()
if save_dir != expected_save_dir:
    raise RuntimeError(
        f"This runner expects meta.save_dir to resolve to {expected_save_dir}, "
        f"got {save_dir}"
    )


def resolve_configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (recipe_root / path).resolve()


paths = {
    "fine-tuned segmentation checkpoint": finetune_checkpoint,
    "stage-2 WavLM checkpoint": stage2_checkpoint,
    "train wav.scp": resolve_configured_path(config["train_dataset"]["args"]["scp_file"]),
    "train RTTM": resolve_configured_path(config["train_dataset"]["args"]["rttm_file"]),
    "train UEM": resolve_configured_path(config["train_dataset"]["args"]["uem_file"]),
    "dev wav.scp": resolve_configured_path(config["validate_dataset"]["args"]["scp_file"]),
    "dev RTTM": resolve_configured_path(config["validate_dataset"]["args"]["rttm_file"]),
    "dev UEM": resolve_configured_path(config["validate_dataset"]["args"]["uem_file"]),
}
missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing configured inputs:\n" + "\n".join(missing))

# apply_pruning.py writes directly to its final filename. Loading the payload here,
# after the process guard above, detects an incomplete or corrupt stage-2 artifact
# before Accelerate creates the stage-3 experiment directory.
stage2_payload = torch.load(stage2_checkpoint, map_location="cpu")
if not isinstance(stage2_payload, Mapping):
    raise TypeError("Stage-2 checkpoint must be a mapping")
stage2_config = stage2_payload.get("config")
stage2_state = stage2_payload.get("state_dict")
if not isinstance(stage2_config, Mapping) or not stage2_config:
    raise RuntimeError("Stage-2 checkpoint has no non-empty config mapping")
if not isinstance(stage2_state, Mapping) or not stage2_state:
    raise RuntimeError("Stage-2 checkpoint has no non-empty state_dict mapping")
invalid_pruning_flags = {
    key: value
    for key, value in stage2_config.items()
    if "prune" in key and value is not False
}
if invalid_pruning_flags:
    raise RuntimeError(
        f"Stage-2 checkpoint still enables pruning flags: {invalid_pruning_flags}"
    )
del stage2_payload, stage2_state

finetune_payload = torch.load(finetune_checkpoint, map_location="cpu")
if isinstance(finetune_payload, Mapping) and isinstance(
    finetune_payload.get("state_dict"), Mapping
):
    finetune_state = finetune_payload["state_dict"]
else:
    finetune_state = finetune_payload
if not isinstance(finetune_state, Mapping) or not finetune_state:
    raise RuntimeError("Fine-tuned segmentation checkpoint has no state dict")
required_prefixes = {
    "wavlm_model.",
    "weight_sum.",
    "proj.",
    "lnorm.",
    "conformer.",
    "classifier.",
}
missing_prefixes = sorted(
    prefix
    for prefix in required_prefixes
    if not any(str(key).startswith(prefix) for key in finetune_state)
)
if missing_prefixes:
    raise RuntimeError(
        "Fine-tuned checkpoint is not a complete segmentation model; missing "
        f"prefixes: {missing_prefixes}"
    )
del finetune_payload, finetune_state

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the configured virtual environment")
if torch.cuda.device_count() < num_processes:
    raise RuntimeError(
        f"Requested {num_processes} processes but PyTorch sees "
        f"{torch.cuda.device_count()} visible CUDA devices"
    )

print(
    {
        "source_configuration": str(config_path),
        "experiment_directory": str(expected_experiment_dir),
        "stage2_wavlm_checkpoint": str(stage2_checkpoint),
        "non_wavlm_initializer": str(finetune_checkpoint),
        "optimizer_small_lr": config["optimizer_small"]["args"]["lr"],
        "optimizer_big_lr": config["optimizer_big"]["args"]["lr"],
        "torch": torch.__version__,
        "visible_cuda_devices": torch.cuda.device_count(),
        "processes": num_processes,
    }
)
PY

stage3_config_workspace="$(mktemp -d "$recipe_root/.stage3-config.XXXXXX")"
stage3_train_conf="$stage3_config_workspace/$stage3_conf_name.toml"
cleanup() {
    rm -f -- "$stage3_train_conf"
    rmdir -- "$stage3_config_workspace" 2>/dev/null || true
}
trap cleanup EXIT
cp -- "$source_train_conf" "$stage3_train_conf"

train_args=(
    launch
    --num_processes "$num_processes"
    --main_process_port "$main_process_port"
    run_dual_opt_continued_training.py
    -C "$stage3_train_conf"
    -M train
    -FT
    --ckpt_path "$finetune_ckpt"
    --pruned_ckpt_path "$stage2_ckpt"
)
if [[ "$resume" == "1" ]]; then
    train_args+=(-R)
fi

echo "Stage 3: starting dual-optimizer diarization training on CUDA devices $gpu_list with $num_processes processes."
echo "WavLM initialization: $stage2_ckpt"
echo "Non-WavLM initialization: $finetune_ckpt"
echo "Stage-3 experiment directory: $stage3_dir"
"$venv_root/bin/accelerate" "${train_args[@]}"

shopt -s nullglob
stage3_model_files=("$stage3_dir"/checkpoints/epoch_[0-9][0-9][0-9][0-9]/pytorch_model.bin)
shopt -u nullglob
if (( ${#stage3_model_files[@]} == 0 )); then
    echo "Stage 3 completed without producing an epoch checkpoint under $stage3_dir" >&2
    exit 1
fi

echo "Stage 3 complete. Experiment directory: $stage3_dir"
