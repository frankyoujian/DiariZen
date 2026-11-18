#!/usr/bin/env bash

# Train the consolidated DiariZen dataset only. This intentionally omits the
# inference, clustering, and dscore evaluation stages from run_stage.sh.

set -euo pipefail
ulimit -n 2048

recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$recipe_root/../.." && pwd)"
venv_root="$project_root/.venv"
train_conf="$recipe_root/conf/wavlm_frozen_conformer_all.toml"

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

"$venv_root/bin/python" - "$train_conf" "$num_processes" <<'PY'
from pathlib import Path
import sys

import toml
import torch

config_path = Path(sys.argv[1]).resolve()
num_processes = int(sys.argv[2])
config = toml.load(config_path)

if config["trainer"]["path"] != "trainer_single_opt.Trainer":
    raise RuntimeError(
        "This runner expects trainer_single_opt.Trainer, got "
        f"{config['trainer']['path']!r}"
    )
if not config["trainer"]["args"].get("freeze_wavlm", False):
    raise RuntimeError("The requested frozen-WavLM configuration must set freeze_wavlm=true")

paths = {
    "WavLM checkpoint": Path(config["model"]["args"]["wavlm_src"]),
    "train wav.scp": Path(config["train_dataset"]["args"]["scp_file"]),
    "train RTTM": Path(config["train_dataset"]["args"]["rttm_file"]),
    "train UEM": Path(config["train_dataset"]["args"]["uem_file"]),
    "dev wav.scp": Path(config["validate_dataset"]["args"]["scp_file"]),
    "dev RTTM": Path(config["validate_dataset"]["args"]["rttm_file"]),
    "dev UEM": Path(config["validate_dataset"]["args"]["uem_file"]),
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
        "checkpoint": str(paths["WavLM checkpoint"]),
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
    run_single_opt.py
    -C "$train_conf"
    -M train
)

if [[ "$resume" == "1" ]]; then
    train_args+=(-R)
elif [[ "$resume" != "0" ]]; then
    echo "RESUME must be 0 or 1, got: $resume" >&2
    exit 1
fi

echo "Starting frozen-WavLM training on CUDA devices $gpu_list with $num_processes processes."
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
