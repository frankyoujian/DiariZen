#!/usr/bin/env bash

# Run stage 2 from run_stage.sh: further-distill the stage-1 pruned model,
# then average the selected checkpoints and extract the student WavLM model.
# Stage 1, diarization training, inference, and scoring are omitted.

set -euo pipefail
ulimit -n 2048

recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$recipe_root/../.." && pwd)"
venv_root="$project_root/.venv"
train_conf="$recipe_root/conf/s80_base_all.toml"

conf_name="${train_conf##*/}"
conf_name="${conf_name%.toml}"
exp_root="$recipe_root/exp"
diarization_dir="$exp_root/$conf_name"
pruned_dir="$diarization_dir/pruned"
pruned_ckpt="$pruned_dir/pytorch_model.bin"
pruned_config="$pruned_dir/pruned_config.json"

further_distill_conf_name="${conf_name}_further_distill"
further_distill_dir="$exp_root/$further_distill_conf_name"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-1134}"
batch_size="${BATCH_SIZE:-16}"
max_epochs="${MAX_EPOCHS:-20}"
avg_ckpt_num="${AVG_CKPT_NUM:-5}"
resume="${RESUME:-0}"
dry_run="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="$gpu_list"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$train_conf" ]]; then
    echo "Missing base training configuration: $train_conf" >&2
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
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer, got: $batch_size" >&2
    exit 1
fi
if [[ ! "$max_epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_EPOCHS must be a positive integer, got: $max_epochs" >&2
    exit 1
fi
if [[ ! "$avg_ckpt_num" =~ ^[1-9][0-9]*$ ]]; then
    echo "AVG_CKPT_NUM must be a positive integer, got: $avg_ckpt_num" >&2
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
    dry_run_config="$recipe_root/<temporary-dir>/$further_distill_conf_name.toml"
    dry_run_train_args=(
        launch
        --num_processes "$num_processes"
        --main_process_port "$main_process_port"
        run_distill_prune.py
        -C "$dry_run_config"
        -M train
        --batch_size "$batch_size"
        --max_epochs "$max_epochs"
        --pruned_ckpt_path "$pruned_ckpt"
    )
    if [[ "$resume" == "1" ]]; then
        dry_run_train_args+=(-R)
    fi

    echo "Dry run only; no directories or configuration files were created."
    echo "Stage 2 prerequisite: $pruned_ckpt and $pruned_config"
    printf 'Dry run stage 2-1 command:'
    printf ' %q' "$venv_root/bin/accelerate" "${dry_run_train_args[@]}"
    printf '\n'

    dry_run_summary="$further_distill_dir/val_metric_summary.lst"
    dry_run_extract_args=(
        apply_pruning.py
        -C "$further_distill_dir/<runtime-config-snapshot.toml>"
        -o "$further_distill_dir/avg_student_ckpt"
        --mode extract
        --avg_ckpt_num "$avg_ckpt_num"
        --val_metric_summary "$dry_run_summary"
    )
    echo "Dry run stage 2-2 first consolidates validation metrics into $dry_run_summary."
    printf 'Dry run stage 2-2 command:'
    printf ' %q' "$venv_root/bin/python" "${dry_run_extract_args[@]}"
    printf '\n'
    exit 0
fi

distill_processes="$(
    pgrep -af '[r]un_distill_prune.py' 2>/dev/null \
        || true
)"
if grep -Fq -- "$further_distill_conf_name.toml" <<< "$distill_processes"; then
    echo "A stage-2 further-distillation process is already running." >&2
    echo "Refusing to launch a duplicate stage-2 job." >&2
    exit 1
fi
if grep -Fq -- "$train_conf" <<< "$distill_processes"; then
    echo "Stage-1 distillation is still running for $train_conf." >&2
    echo "Refusing to start stage 2 until the stage-1 process has exited." >&2
    exit 1
fi

# pruned_config.json is written only after apply_pruning.py has finished writing
# and verifying pytorch_model.bin, so requiring both avoids consuming a partial
# stage-1 artifact while stage 1-2 is still running.
if [[ ! -s "$pruned_ckpt" || ! -s "$pruned_config" ]]; then
    echo "Stage 2 requires completed stage-1 pruning artifacts:" >&2
    echo "  $pruned_ckpt" >&2
    echo "  $pruned_config" >&2
    echo "Wait for run_stage1_only.sh to exit successfully before starting stage 2." >&2
    exit 1
fi

if [[ "$resume" == "0" && -e "$further_distill_dir" ]]; then
    echo "Experiment directory already exists: $further_distill_dir" >&2
    echo "Set RESUME=1 to continue it, or move it aside before starting a fresh run." >&2
    exit 1
fi
if [[ "$resume" == "1" ]]; then
    shopt -s nullglob
    resume_model_files=("$further_distill_dir"/checkpoints/epoch_[0-9][0-9][0-9][0-9]/pytorch_model.bin)
    shopt -u nullglob
    if [[ ! -d "$further_distill_dir/tb_log" || ${#resume_model_files[@]} -eq 0 \
        || ! -f "${resume_model_files[0]:-}" ]]; then
        echo "RESUME=1 requires tb_log and at least one epoch checkpoint under $further_distill_dir" >&2
        exit 1
    fi
fi

cd "$recipe_root"

"$venv_root/bin/python" - \
    "$train_conf" \
    "$pruned_ckpt" \
    "$pruned_config" \
    "$num_processes" \
    "$exp_root" \
    "$batch_size" \
    "$max_epochs" <<'PY'
import json
from pathlib import Path
import sys

import toml
import torch

(
    config_value,
    pruned_checkpoint_value,
    pruned_config_value,
    num_processes_value,
    expected_save_dir_value,
    batch_size_value,
    max_epochs_value,
) = sys.argv[1:]

config_path = Path(config_value).resolve()
pruned_checkpoint = Path(pruned_checkpoint_value).resolve()
pruned_config_path = Path(pruned_config_value).resolve()
num_processes = int(num_processes_value)
expected_save_dir = Path(expected_save_dir_value).resolve()
batch_size = int(batch_size_value)
max_epochs = int(max_epochs_value)
recipe_root = config_path.parent.parent
config = toml.load(config_path)

if config["trainer"]["path"] != "trainer_distill_prune.Trainer":
    raise RuntimeError(
        "This runner expects trainer_distill_prune.Trainer, got "
        f"{config['trainer']['path']!r}"
    )
if config["finetune"].get("finetune", False):
    raise RuntimeError("Stage 2 expects finetune=false")
if not config["trainer"]["args"].get("use_reg", False):
    raise RuntimeError(
        "The base stage-1 configuration must have use_reg=true; "
        "run_distill_prune.py disables it for further distillation"
    )

required_pruning_keys = {
    "target_sparsity",
    "pre_train_epochs",
    "sparsity_warmup_epochs",
}
missing_pruning_keys = sorted(
    required_pruning_keys - config["trainer"]["args"].keys()
)
if missing_pruning_keys:
    raise RuntimeError(
        "Base configuration is missing keys required by --pruned_ckpt_path: "
        + ", ".join(missing_pruning_keys)
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
    "teacher checkpoint": resolve_configured_path(config["model"]["args"]["teacher_ckpt"]),
    "stage-1 pruned checkpoint": pruned_checkpoint,
    "stage-1 pruned metadata": pruned_config_path,
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

with pruned_config_path.open("r", encoding="utf-8") as handle:
    metadata = json.load(handle)
if not isinstance(metadata, dict) or not metadata:
    raise RuntimeError(f"Invalid stage-1 pruning metadata: {pruned_config_path}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the configured virtual environment")
if torch.cuda.device_count() < num_processes:
    raise RuntimeError(
        f"Requested {num_processes} processes but PyTorch sees "
        f"{torch.cuda.device_count()} visible CUDA devices"
    )

print(
    {
        "base_configuration": str(config_path),
        "experiment_directory": str(expected_save_dir / f"{config_path.stem}_further_distill"),
        "teacher_checkpoint": str(paths["teacher checkpoint"]),
        "pruned_student_checkpoint": str(pruned_checkpoint),
        "batch_size_per_gpu": batch_size,
        "max_epochs": max_epochs,
        "torch": torch.__version__,
        "visible_cuda_devices": torch.cuda.device_count(),
        "processes": num_processes,
    }
)
PY

stage2_config_workspace="$(mktemp -d "$recipe_root/.stage2-config.XXXXXX")"
stage2_train_conf="$stage2_config_workspace/$further_distill_conf_name.toml"
cleanup() {
    rm -f -- "$stage2_train_conf"
    rmdir -- "$stage2_config_workspace" 2>/dev/null || true
}
trap cleanup EXIT
cp -- "$train_conf" "$stage2_train_conf"

train_args=(
    launch
    --num_processes "$num_processes"
    --main_process_port "$main_process_port"
    run_distill_prune.py
    -C "$stage2_train_conf"
    -M train
    --batch_size "$batch_size"
    --max_epochs "$max_epochs"
    --pruned_ckpt_path "$pruned_ckpt"
)
if [[ "$resume" == "1" ]]; then
    train_args+=(-R)
fi

echo "Stage 2-1: starting further distillation on CUDA devices $gpu_list with $num_processes processes."
echo "Stage-1 pruned model: $pruned_ckpt"
echo "Stage-2 experiment directory: $further_distill_dir"
"$venv_root/bin/accelerate" "${train_args[@]}"

echo "Stage 2-2: selecting checkpoints and extracting the averaged student model."
shopt -s nullglob
config_files=("$further_distill_dir"/config__*.toml)
log_files=("$further_distill_dir"/"${further_distill_conf_name}"_*.log)
shopt -u nullglob

if (( ${#config_files[@]} == 0 )); then
    echo "No runtime configuration snapshot found under $further_distill_dir" >&2
    exit 1
fi
if (( ${#log_files[@]} == 0 )); then
    echo "No training log found under $further_distill_dir" >&2
    exit 1
fi

further_distill_config="${config_files[0]}"
for candidate in "${config_files[@]:1}"; do
    if [[ "$candidate" > "$further_distill_config" ]]; then
        further_distill_config="$candidate"
    fi
done

val_metric_summary="$further_distill_dir/val_metric_summary.lst"
"$venv_root/bin/python" - \
    "$val_metric_summary" \
    "$further_distill_dir/checkpoints" \
    "$further_distill_config" \
    "$pruned_ckpt" \
    "$batch_size" \
    "$max_epochs" \
    "${log_files[@]}" <<'PY'
from pathlib import Path
import re
import sys

import toml

summary_path = Path(sys.argv[1])
checkpoints_dir = Path(sys.argv[2])
runtime_config_path = Path(sys.argv[3]).resolve()
pruned_checkpoint = Path(sys.argv[4]).resolve()
expected_batch_size = int(sys.argv[5])
expected_max_epochs = int(sys.argv[6])
log_paths = sorted(Path(value) for value in sys.argv[7:])

runtime_config = toml.load(runtime_config_path)
model_args = runtime_config["model"]["args"]
trainer_args = runtime_config["trainer"]["args"]
dataloader_args = runtime_config["train_dataset"]["dataloader"]

runtime_student_checkpoint = Path(model_args["student_ckpt"]).expanduser()
if not runtime_student_checkpoint.is_absolute():
    runtime_student_checkpoint = (runtime_config_path.parent / runtime_student_checkpoint).resolve()
else:
    runtime_student_checkpoint = runtime_student_checkpoint.resolve()

if runtime_student_checkpoint != pruned_checkpoint:
    raise RuntimeError(
        "Runtime configuration references the wrong pruned student checkpoint: "
        f"{runtime_student_checkpoint} (expected {pruned_checkpoint})"
    )
if str(model_args.get("pruning_units", "")).strip():
    raise RuntimeError("Runtime configuration still enables pruning units")
if trainer_args.get("use_reg", True):
    raise RuntimeError("Runtime configuration still enables pruning regularization")
if not trainer_args.get("further_distill", False):
    raise RuntimeError("Runtime configuration is not marked for further distillation")
if trainer_args.get("max_epochs") != expected_max_epochs:
    raise RuntimeError(
        f"Runtime max_epochs is {trainer_args.get('max_epochs')}, "
        f"expected {expected_max_epochs}"
    )
if dataloader_args.get("batch_size") != expected_batch_size:
    raise RuntimeError(
        f"Runtime train batch size is {dataloader_args.get('batch_size')}, "
        f"expected {expected_batch_size}"
    )
if runtime_config_path.parent != checkpoints_dir.parent.resolve():
    raise RuntimeError(
        f"Runtime configuration must be stored beside checkpoints in "
        f"{checkpoints_dir.parent.resolve()}, got {runtime_config_path}"
    )

metric_by_epoch = {}
metric_source_by_epoch = {}
for log_path in log_paths:
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "Validation Loss | Loss_distill"
        if marker not in raw_line:
            continue
        message = raw_line[raw_line.index(marker):].strip()
        match = re.search(r"\bon epoch (\d+):", message)
        if match is None:
            raise RuntimeError(f"Cannot parse validation epoch from {log_path}: {message}")
        epoch = int(match.group(1))
        previous = metric_by_epoch.get(epoch)
        if previous is not None and previous != message:
            raise RuntimeError(
                f"Conflicting validation metrics for epoch {epoch} in "
                f"{metric_source_by_epoch[epoch]} and {log_path}"
            )
        metric_by_epoch[epoch] = message
        metric_source_by_epoch[epoch] = log_path

if not metric_by_epoch:
    raise RuntimeError(f"No validation metrics found in: {', '.join(map(str, log_paths))}")

missing_checkpoints = [
    checkpoints_dir / f"epoch_{epoch:04d}" / "pytorch_model.bin"
    for epoch in sorted(metric_by_epoch)
    if not (checkpoints_dir / f"epoch_{epoch:04d}" / "pytorch_model.bin").is_file()
]
if missing_checkpoints:
    raise FileNotFoundError(
        "Validation metrics reference missing checkpoints:\n"
        + "\n".join(map(str, missing_checkpoints))
    )

summary_text = "\n".join(metric_by_epoch[epoch] for epoch in sorted(metric_by_epoch)) + "\n"
temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
temporary_path.write_text(summary_text, encoding="utf-8")
temporary_path.replace(summary_path)
print(f"Saved {len(metric_by_epoch)} validation metrics to {summary_path}")
PY

student_output_dir="$further_distill_dir/avg_student_ckpt"
extract_args=(
    apply_pruning.py
    -C "$further_distill_config"
    -o "$student_output_dir"
    --mode extract
    --avg_ckpt_num "$avg_ckpt_num"
    --val_metric_summary "$val_metric_summary"
)

echo "Further-distillation configuration snapshot: $further_distill_config"
echo "Training logs: ${#log_files[@]} file(s)"
"$venv_root/bin/python" "${extract_args[@]}"

if [[ ! -s "$student_output_dir/pytorch_model.bin" ]]; then
    echo "Stage 2-2 did not produce $student_output_dir/pytorch_model.bin" >&2
    exit 1
fi

echo "Stage 2 complete. Averaged student model: $student_output_dir/pytorch_model.bin"
