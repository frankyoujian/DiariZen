#!/usr/bin/env bash

# Run stage 1 from run_stage.sh: first train the distillation/pruning model,
# then average the selected checkpoints and materialize the pruned model.
# Further distillation, diarization training, inference, and scoring are omitted.

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

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-1134}"
avg_ckpt_num="${AVG_CKPT_NUM:-5}"
resume="${RESUME:-0}"
dry_run="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="$gpu_list"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

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

if [[ "$resume" == "0" && -e "$diarization_dir" ]]; then
    echo "Experiment directory already exists: $diarization_dir" >&2
    echo "Set RESUME=1 to continue it, or move it aside before starting a fresh run." >&2
    exit 1
fi
if [[ "$resume" == "1" ]]; then
    shopt -s nullglob
    resume_model_files=("$diarization_dir"/checkpoints/epoch_[0-9][0-9][0-9][0-9]/pytorch_model.bin)
    shopt -u nullglob
    if [[ ! -d "$diarization_dir/tb_log" || ${#resume_model_files[@]} -eq 0 \
        || ! -f "${resume_model_files[0]:-}" ]]; then
        echo "RESUME=1 requires tb_log and at least one epoch checkpoint under $diarization_dir" >&2
        exit 1
    fi
fi

cd "$recipe_root"

"$venv_root/bin/python" - "$train_conf" "$num_processes" "$exp_root" <<'PY'
from pathlib import Path
import sys

import toml
import torch

config_path = Path(sys.argv[1]).resolve()
num_processes = int(sys.argv[2])
expected_save_dir = Path(sys.argv[3]).resolve()
recipe_root = config_path.parent.parent
config = toml.load(config_path)

if config["trainer"]["path"] != "trainer_distill_prune.Trainer":
    raise RuntimeError(
        "This runner expects trainer_distill_prune.Trainer, got "
        f"{config['trainer']['path']!r}"
    )
if config["finetune"].get("finetune", False):
    raise RuntimeError("Stage 1-1 expects finetune=false")
if not config["trainer"]["args"].get("use_reg", False):
    raise RuntimeError("Stage 1-1 distillation/pruning requires use_reg=true")

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
    "student checkpoint": resolve_configured_path(config["model"]["args"]["student_ckpt"]),
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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the configured virtual environment")
if torch.cuda.device_count() < num_processes:
    raise RuntimeError(
        f"Requested {num_processes} processes but PyTorch sees "
        f"{torch.cuda.device_count()} visible CUDA devices"
    )

print(
    {
        "configuration": str(config_path),
        "experiment_directory": str(expected_save_dir / config_path.stem),
        "teacher_checkpoint": str(paths["teacher checkpoint"]),
        "student_checkpoint": str(paths["student checkpoint"]),
        "target_sparsity": config["trainer"]["args"]["target_sparsity"],
        "distill_lr": config["optimizer"]["args"]["distill_lr"],
        "reg_lr": config["optimizer"]["args"]["reg_lr"],
        "torch": torch.__version__,
        "visible_cuda_devices": torch.cuda.device_count(),
        "processes": num_processes,
    }
)
PY

train_args=(
    launch
    --num_processes "$num_processes"
    --main_process_port "$main_process_port"
    run_distill_prune.py
    -C "$train_conf"
    -M train
)
if [[ "$resume" == "1" ]]; then
    train_args+=(-R)
fi

if [[ "$dry_run" == "1" ]]; then
    printf 'Dry run stage 1-1 command:'
    printf ' %q' "$venv_root/bin/accelerate" "${train_args[@]}"
    printf '\n'

    dry_run_config="$diarization_dir/<input-config-snapshot.toml>"
    dry_run_summary="$diarization_dir/val_metric_summary.lst"
    dry_run_prune_args=(
        apply_pruning.py
        -C "$dry_run_config"
        -o "$diarization_dir/pruned"
        --mode prune
        --avg_ckpt_num "$avg_ckpt_num"
        --val_metric_summary "$dry_run_summary"
    )
    echo "Dry run stage 1-2 first consolidates validation metrics from the experiment logs into $dry_run_summary."
    printf 'Dry run stage 1-2 command:'
    printf ' %q' "$venv_root/bin/python" "${dry_run_prune_args[@]}"
    printf '\n'
    exit 0
fi

mkdir -p -- "$diarization_dir"
pruning_config="$(mktemp --suffix=.toml "$diarization_dir/.stage1-prune-config.XXXXXX")"
cleanup() {
    rm -f -- "$pruning_config"
}
trap cleanup EXIT
cp -- "$train_conf" "$pruning_config"

echo "Stage 1-1: starting distillation/pruning training on CUDA devices $gpu_list with $num_processes processes."
echo "Training-time development validation remains enabled by the TOML configuration."
"$venv_root/bin/accelerate" "${train_args[@]}"

echo "Stage 1-2: selecting checkpoints and materializing the pruned model."
shopt -s nullglob
log_files=("$diarization_dir"/"${conf_name}"_*.log)
shopt -u nullglob

if (( ${#log_files[@]} == 0 )); then
    echo "No training log found under $diarization_dir" >&2
    exit 1
fi

val_metric_summary="$diarization_dir/val_metric_summary.lst"
"$venv_root/bin/python" - \
    "$val_metric_summary" \
    "$diarization_dir/checkpoints" \
    "${log_files[@]}" <<'PY'
from pathlib import Path
import re
import sys

summary_path = Path(sys.argv[1])
checkpoints_dir = Path(sys.argv[2])
log_paths = sorted(Path(value) for value in sys.argv[3:])

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

pruned_dir="$diarization_dir/pruned"
prune_args=(
    apply_pruning.py
    -C "$pruning_config"
    -o "$pruned_dir"
    --mode prune
    --avg_ckpt_num "$avg_ckpt_num"
    --val_metric_summary "$val_metric_summary"
)

echo "Pruning configuration snapshot: $pruning_config"
echo "Training logs: ${#log_files[@]} file(s)"
"$venv_root/bin/python" "${prune_args[@]}"

if [[ ! -s "$pruned_dir/pytorch_model.bin" || ! -s "$pruned_dir/pruned_config.json" ]]; then
    echo "Stage 1-2 did not produce the expected files under $pruned_dir" >&2
    exit 1
fi

echo "Stage 1 complete. Pruned model: $pruned_dir/pytorch_model.bin"
