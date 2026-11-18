#!/usr/bin/env python3
"""Average five selected DiariZen model checkpoints into one state dictionary.

Example:
    python average_selected_checkpoints.py \
        --checkpoint-root /path/to/checkpoints \
        --epochs 80 85 90 78 79 \
        --output /path/to/averaged_checkpoint

The averaged model is written to ``<output>/pytorch_model.bin``. Only model
weights are averaged; optimizer, scheduler, and trainer state files are ignored.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from collections.abc import MutableMapping, Sequence
from pathlib import Path

import torch


MODEL_FILENAME = "pytorch_model.bin"


def parse_epoch(value: str) -> int:
    """Parse a non-negative epoch number supplied on the command line."""
    try:
        epoch = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid epoch {value!r}: expected an integer") from exc

    if epoch < 0:
        raise argparse.ArgumentTypeError(f"invalid epoch {value!r}: expected a non-negative integer")
    return epoch


def checkpoint_paths(checkpoint_root: Path, epochs: Sequence[int]) -> list[Path]:
    """Resolve and validate five epoch checkpoint paths."""
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint root is not a directory: {checkpoint_root}")
    if len(epochs) != 5:
        raise ValueError(f"exactly five epochs must be provided, got {len(epochs)}")
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"duplicate epochs are not allowed: {list(epochs)}")

    paths = [checkpoint_root / f"epoch_{epoch:04d}" / MODEL_FILENAME for epoch in epochs]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing model checkpoint(s):\n{formatted}")
    return paths


def load_state_dict(path: Path) -> MutableMapping[str, torch.Tensor]:
    """Load a plain PyTorch model state dictionary on CPU."""
    try:
        state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases that lack weights_only.
        state_dict = torch.load(path, map_location=torch.device("cpu"))

    if not isinstance(state_dict, MutableMapping):
        raise TypeError(f"checkpoint is not a mutable state dictionary: {path}")
    if not state_dict:
        raise ValueError(f"checkpoint contains an empty state dictionary: {path}")

    invalid_keys = [key for key, value in state_dict.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid_keys:
        preview = ", ".join(repr(key) for key in invalid_keys[:5])
        raise TypeError(f"checkpoint contains non-tensor state entries ({preview}): {path}")
    return state_dict


def accumulator_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Choose a stable accumulation dtype while preserving common FP32 behavior."""
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return tensor.dtype


def average_state_dicts(paths: Sequence[Path]) -> MutableMapping[str, torch.Tensor]:
    """Average compatible checkpoint state dictionaries using bounded memory."""
    averaged_state = load_state_dict(paths[0])
    expected_keys = list(averaged_state.keys())
    expected_key_set = set(expected_keys)
    output_dtypes: dict[str, torch.dtype] = {}
    differing_nonfloating_keys: set[str] = set()

    for key in expected_keys:
        tensor = averaged_state[key]
        output_dtypes[key] = tensor.dtype
        if tensor.is_floating_point() or tensor.is_complex():
            averaged_state[key] = tensor.to(dtype=accumulator_dtype(tensor))

    for path in paths[1:]:
        state = load_state_dict(path)
        actual_key_set = set(state.keys())
        if actual_key_set != expected_key_set:
            missing = sorted(expected_key_set - actual_key_set)
            unexpected = sorted(actual_key_set - expected_key_set)
            raise ValueError(
                f"state dictionary keys differ in {path}; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        for key in expected_keys:
            candidate = state[key]
            averaged = averaged_state[key]
            expected_dtype = output_dtypes[key]

            if candidate.shape != averaged.shape:
                raise ValueError(
                    f"shape mismatch for {key!r} in {path}: "
                    f"expected {tuple(averaged.shape)}, got {tuple(candidate.shape)}"
                )
            if candidate.dtype != expected_dtype:
                raise ValueError(
                    f"dtype mismatch for {key!r} in {path}: "
                    f"expected {expected_dtype}, got {candidate.dtype}"
                )

            if candidate.is_floating_point() or candidate.is_complex():
                averaged.add_(candidate.to(dtype=averaged.dtype))
            elif not torch.equal(averaged, candidate):
                differing_nonfloating_keys.add(key)

        del state

    count = len(paths)
    for key in expected_keys:
        tensor = averaged_state[key]
        if tensor.is_floating_point() or tensor.is_complex():
            tensor.div_(count)
            target_dtype = output_dtypes[key]
            if tensor.dtype != target_dtype:
                averaged_state[key] = tensor.to(dtype=target_dtype)

    if differing_nonfloating_keys:
        formatted_keys = ", ".join(sorted(differing_nonfloating_keys))
        warnings.warn(
            "non-floating state entries differed across checkpoints; "
            f"kept values from the first checkpoint for: {formatted_keys}",
            stacklevel=2,
        )

    return averaged_state


def save_state_dict_atomic(
    state_dict: MutableMapping[str, torch.Tensor],
    output_path: Path,
    *,
    force: bool,
) -> None:
    """Save a state dictionary atomically without overwriting by default."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists (use --force to overwrite): {output_path}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        torch.save(state_dict, temporary_path)
        if output_path.exists() and not force:
            raise FileExistsError(f"output appeared while saving; refusing to overwrite: {output_path}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average five selected DiariZen epoch checkpoints into <output>/pytorch_model.bin."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Directory containing epoch_NNNN checkpoint folders.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=parse_epoch,
        required=True,
        help="Exactly five distinct epoch indices, for example: --epochs 80 85 90 78 79.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"Output folder; the averaged state dictionary is saved as {MODEL_FILENAME}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite <output>/pytorch_model.bin if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_directory = args.output.expanduser().resolve()
    paths = checkpoint_paths(checkpoint_root, args.epochs)
    output_path = output_directory / MODEL_FILENAME

    if output_path in {path.resolve() for path in paths}:
        raise ValueError("output checkpoint must not overwrite one of the selected input checkpoints")

    print("Averaging checkpoints:")
    for path in paths:
        print(f"  {path}")

    averaged_state = average_state_dicts(paths)
    save_state_dict_atomic(averaged_state, output_path, force=args.force)
    print(f"Saved averaged checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
