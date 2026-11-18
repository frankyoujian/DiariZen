#!/usr/bin/env bash
set -euo pipefail

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${runtime_dir}/.." && pwd)"
python="${repo_root}/.venv/bin/python"

if [[ ! -x "${python}" ]]; then
  echo "Virtual environment not found: ${python}" >&2
  exit 1
fi

"${python}" "${runtime_dir}/export_segmentation_onnx.py" "$@"
"${python}" "${runtime_dir}/validate_segmentation_onnx.py"
