#!/usr/bin/env bash

# Fine-tune the consolidated DiariZen dataset with separate WavLM and
# non-WavLM optimizers. Inference, clustering, and scoring are omitted.

set -euo pipefail
ulimit -n 2048

recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$recipe_root/../.." && pwd)"
venv_root="$project_root/.venv"
train_conf="$recipe_root/conf/wavlm_updated_conformer_all.toml"
finetune_ckpt="${FINETUNE_CKPT:-$recipe_root/exp_ReduceLROnPlataeu/wavlm_frozen_conformer_all/checkpoints/average_64_65_69_70_72/pytorch_model.bin}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
num_processes="${NUM_PROCESSES:-8}"
main_process_port="${MAIN_PROCESS_PORT:-1134}"
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

cd "$recipe_root"

"$venv_root/bin/python" - "$train_conf" "$finetune_ckpt" "$num_processes" <<'PY'
from pathlib import Path
import sys

import toml
import torch

config_path = Path(sys.argv[1]).resolve()
finetune_ckpt = Path(sys.argv[2]).resolve()
num_processes = int(sys.argv[3])
recipe_root = config_path.parent.parent
config = toml.load(config_path)

if config["trainer"]["path"] != "trainer_dual_opt.Trainer":
    raise RuntimeError(
        "This runner expects trainer_dual_opt.Trainer, got "
        f"{config['trainer']['path']!r}"
    )
if config["trainer"]["args"].get("freeze_wavlm", False):
    raise RuntimeError("Dual-optimizer fine-tuning requires freeze_wavlm=false")


def resolve_configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (recipe_root / path).resolve()


paths = {
    "WavLM bootstrap checkpoint": resolve_configured_path(config["model"]["args"]["wavlm_src"]),
    "fine-tune checkpoint": finetune_ckpt,
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
        "WavLM bootstrap": str(paths["WavLM bootstrap checkpoint"]),
        "fine-tune checkpoint": str(paths["fine-tune checkpoint"]),
        "optimizer_small_lr": config["optimizer_small"]["args"]["lr"],
        "optimizer_big_lr": config["optimizer_big"]["args"]["lr"],
        "lr_decay_factor": config["trainer"]["args"].get("lr_decay_factor"),
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
    run_dual_opt.py
    -C "$train_conf"
    -M train
    -FT
    --ckpt_path "$finetune_ckpt"
)

if [[ "$resume" == "1" ]]; then
    train_args+=(-R)
elif [[ "$resume" != "0" ]]; then
    echo "RESUME must be 0 or 1, got: $resume" >&2
    exit 1
fi

echo "Starting dual-optimizer fine-tuning on CUDA devices $gpu_list with $num_processes processes."
echo "Initial model: $finetune_ckpt"
echo "Training-time development validation remains enabled by the TOML configuration."
if [[ "$dry_run" == "1" ]]; then
    printf 'Dry run command:'
    printf ' %q' "$venv_root/bin/accelerate" "${train_args[@]}"
    printf '\n'
    exit 0
elif [[ "$dry_run" != "0" ]]; then
    echo "DRY_RUN must be 0 or 1, got: $dry_run" >&2
    exit 1
fi

exec "$venv_root/bin/accelerate" "${train_args[@]}"
