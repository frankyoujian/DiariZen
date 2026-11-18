#!/usr/bin/env python3
"""Extract a standalone WavLM checkpoint from a diarization checkpoint."""

import argparse
import copy
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from diarizen.models.module.wav2vec2.model import wav2vec2_model


DEFAULT_BOOTSTRAP = (
    Path(__file__).resolve().parents[2] / "pretrained" / "WavLM-Base+.diarizen.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract WavLM weights from an already-averaged full diarization "
            "checkpoint and package them as a standalone DiariZen WavLM checkpoint."
        ),
        epilog="Only load checkpoint files from trusted sources.",
    )
    parser.add_argument(
        "-i",
        "--input",
        "--input-checkpoint",
        dest="input_checkpoint",
        type=Path,
        required=True,
        help="Full diarization checkpoint (raw state dict or state_dict wrapper).",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-checkpoint",
        dest="output_checkpoint",
        type=Path,
        required=True,
        help="Destination standalone WavLM checkpoint.",
    )
    parser.add_argument(
        "-b",
        "--bootstrap",
        "--bootstrap-checkpoint",
        dest="bootstrap_checkpoint",
        type=Path,
        default=DEFAULT_BOOTSTRAP,
        help=(
            "Standalone WavLM checkpoint supplying the architecture config and "
            "reference state-dict schema (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="wavlm_model.",
        help=(
            "Prefix to select and remove from source keys "
            "(default: %(default)r; use an empty string for an unprefixed state dict)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace the output if it already exists.",
    )
    return parser.parse_args()


def load_state_dict(path: Path, label: str) -> tuple[Mapping[str, Any], Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"{label} checkpoint must be a mapping, got {type(checkpoint).__name__}: {path}"
        )

    wrapped_state_dict = checkpoint.get("state_dict")
    if isinstance(wrapped_state_dict, Mapping):
        return wrapped_state_dict, checkpoint
    return checkpoint, checkpoint


def preview(keys: list[str], limit: int = 10) -> str:
    if not keys:
        return "none"
    shown = ", ".join(keys[:limit])
    if len(keys) > limit:
        shown += f", ... (+{len(keys) - limit} more)"
    return shown


def paths_refer_to_same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def save_atomically(payload: dict[str, Any], output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output) and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    fd, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_path = Path(temporary_name)

    try:
        torch.save(payload, temporary_path)
        if force:
            os.replace(temporary_path, output)
        else:
            try:
                # Same-directory hard linking provides atomic no-clobber creation.
                os.link(temporary_path, output)
            except FileExistsError:
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {output}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    input_path = args.input_checkpoint.expanduser().resolve()
    bootstrap_path = args.bootstrap_checkpoint.expanduser().resolve()
    output_argument = args.output_checkpoint.expanduser()
    output_path = output_argument.parent.resolve() / output_argument.name

    for label, path in (
        ("Input", input_path),
        ("Bootstrap", bootstrap_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint is not a file: {path}")

    if paths_refer_to_same_file(output_path, input_path):
        raise ValueError("Output must not overwrite the input checkpoint")
    if paths_refer_to_same_file(output_path, bootstrap_path):
        raise ValueError("Output must not overwrite the bootstrap checkpoint")

    source, _ = load_state_dict(input_path, "Input")
    reference, bootstrap = load_state_dict(bootstrap_path, "Bootstrap")

    non_string_source_keys = sorted(
        repr(key) for key in source if not isinstance(key, str)
    )
    non_string_reference_keys = sorted(
        repr(key) for key in reference if not isinstance(key, str)
    )
    if non_string_source_keys or non_string_reference_keys:
        raise TypeError(
            "State-dict keys must be strings. "
            f"Input keys: {preview(non_string_source_keys)}; "
            f"reference keys: {preview(non_string_reference_keys)}"
        )

    if not isinstance(bootstrap.get("config"), Mapping):
        raise KeyError(
            f"Bootstrap checkpoint must contain a mapping-valued 'config': {bootstrap_path}"
        )
    if not isinstance(bootstrap.get("state_dict"), Mapping):
        raise KeyError(
            f"Bootstrap checkpoint must contain a mapping-valued 'state_dict': {bootstrap_path}"
        )

    prefix = args.prefix
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if not key.startswith(prefix):
            continue
        stripped_key = key[len(prefix) :]
        if stripped_key in state_dict:
            raise RuntimeError(f"Duplicate extracted key after removing prefix: {stripped_key}")
        state_dict[stripped_key] = value

    if not state_dict:
        raise RuntimeError(
            f"No source keys matched prefix {prefix!r}; check --prefix and the input checkpoint"
        )

    non_tensor_source = sorted(
        key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)
    )
    non_tensor_reference = sorted(
        key for key, value in reference.items() if not isinstance(value, torch.Tensor)
    )
    if non_tensor_source or non_tensor_reference:
        raise TypeError(
            "State dictionaries must contain only tensors. "
            f"Extracted non-tensors: {preview(non_tensor_source)}; "
            f"reference non-tensors: {preview(non_tensor_reference)}"
        )

    extracted_keys = set(state_dict)
    reference_keys = set(reference)
    missing = sorted(reference_keys - extracted_keys)
    unexpected = sorted(extracted_keys - reference_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Extracted WavLM keys do not match the bootstrap model. "
            f"Missing ({len(missing)}): {preview(missing)}; "
            f"unexpected ({len(unexpected)}): {preview(unexpected)}"
        )

    mismatched = sorted(
        key
        for key in state_dict
        if state_dict[key].shape != reference[key].shape
        or state_dict[key].dtype != reference[key].dtype
    )
    if mismatched:
        details = "; ".join(
            f"{key}: extracted={tuple(state_dict[key].shape)}/{state_dict[key].dtype}, "
            f"reference={tuple(reference[key].shape)}/{reference[key].dtype}"
            for key in mismatched[:10]
        )
        if len(mismatched) > 10:
            details += f"; ... (+{len(mismatched) - 10} more)"
        raise RuntimeError(f"Shape/dtype mismatches: {details}")

    config = copy.deepcopy(bootstrap["config"])
    model = wav2vec2_model(**config)
    model.load_state_dict(state_dict, strict=True)

    save_atomically(
        {
            "config": config,
            "state_dict": state_dict,
        },
        output_path,
        args.force,
    )

    parameter_count = sum(tensor.numel() for tensor in state_dict.values())
    print(f"Input: {input_path}")
    print(f"Bootstrap: {bootstrap_path}")
    print(f"Prefix removed: {prefix!r}")
    print(f"Extracted: {len(state_dict)} tensors, {parameter_count:,} values")
    print(f"Saved standalone WavLM checkpoint: {output_path}")


if __name__ == "__main__":
    main()
