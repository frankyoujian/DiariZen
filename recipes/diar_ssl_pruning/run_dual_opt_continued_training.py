# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import toml
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from diarizen.logger import init_logging_logger
from diarizen.utils import instantiate

from dataset import _collate_fn


WAVLM_PREFIX = "wavlm_model."


def _format_keys(keys, limit=10):
    keys = sorted(keys)
    preview = ", ".join(keys[:limit])
    if len(keys) > limit:
        preview += f", ... ({len(keys)} total)"
    return preview


def _extract_state_dict(checkpoint, checkpoint_path):
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint {checkpoint_path} must contain a mapping, got "
            f"{type(checkpoint).__name__}"
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError(
            f"Checkpoint {checkpoint_path} does not contain a non-empty state dict"
        )

    invalid_keys = [key for key in state_dict if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError(f"Checkpoint {checkpoint_path} contains non-string state keys")

    non_tensor_keys = [
        key for key, value in state_dict.items() if not torch.is_tensor(value)
    ]
    if non_tensor_keys:
        raise TypeError(
            f"Checkpoint {checkpoint_path} contains non-tensor state values: "
            f"{_format_keys(non_tensor_keys)}"
        )

    module_prefixed = [key.startswith("module.") for key in state_dict]
    if any(module_prefixed):
        if not all(module_prefixed):
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} mixes module-prefixed and unprefixed keys"
            )
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }

    return dict(state_dict)


def load_non_wavlm_checkpoint(model, checkpoint_path):
    """Initialize every segmentation-model tensor except ``wavlm_model.*``.

    The WavLM module is already constructed from the stage-2 pruned checkpoint.
    This loader therefore enforces exact key and shape compatibility for all
    remaining modules while deliberately leaving every WavLM tensor untouched.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = _extract_state_dict(checkpoint, checkpoint_path)
    target_state = model.state_dict()

    target_wavlm_keys = {
        key for key in target_state if key.startswith(WAVLM_PREFIX)
    }
    expected_non_wavlm_keys = set(target_state) - target_wavlm_keys
    source_wavlm_keys = {
        key for key in source_state if key.startswith(WAVLM_PREFIX)
    }
    source_non_wavlm = {
        key: value
        for key, value in source_state.items()
        if not key.startswith(WAVLM_PREFIX)
    }

    missing_keys = expected_non_wavlm_keys - set(source_non_wavlm)
    unexpected_keys = set(source_non_wavlm) - expected_non_wavlm_keys
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing: {_format_keys(missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected: {_format_keys(unexpected_keys)}")
        raise RuntimeError(
            "Non-WavLM checkpoint keys do not exactly match the segmentation model ("
            + "; ".join(details)
            + ")"
        )

    shape_mismatches = [
        (
            key,
            tuple(source_non_wavlm[key].shape),
            tuple(target_state[key].shape),
        )
        for key in sorted(expected_non_wavlm_keys)
        if source_non_wavlm[key].shape != target_state[key].shape
    ]
    if shape_mismatches:
        preview = ", ".join(
            f"{key}: checkpoint{source_shape} != model{target_shape}"
            for key, source_shape, target_shape in shape_mismatches[:10]
        )
        if len(shape_mismatches) > 10:
            preview += f", ... ({len(shape_mismatches)} total)"
        raise RuntimeError(f"Non-WavLM tensor shape mismatches: {preview}")

    incompatible = model.load_state_dict(source_non_wavlm, strict=False)
    actual_missing = set(incompatible.missing_keys)
    actual_unexpected = set(incompatible.unexpected_keys)
    if actual_missing != target_wavlm_keys or actual_unexpected:
        raise RuntimeError(
            "Selective checkpoint load produced unexpected incompatibilities: "
            f"missing={_format_keys(actual_missing)}, "
            f"unexpected={_format_keys(actual_unexpected)}"
        )

    return {
        "loaded_non_wavlm_tensors": len(source_non_wavlm),
        "preserved_wavlm_tensors": len(target_wavlm_keys),
        "ignored_source_wavlm_tensors": len(source_wavlm_keys),
    }


def run(config, resume):
    init_logging_logger(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    # The constructor initializes wavlm_model from the stage-2 pruned checkpoint.
    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    if config["finetune"]["finetune"]:
        finetune_checkpoint = config["finetune"]["ckpt_dir"]
        if resume:
            accelerator.print(
                "Resume requested: skipping non-WavLM initialization because the "
                "stage-3 checkpoint will restore the complete model."
            )
        else:
            accelerator.print(
                "Initializing non-WavLM segmentation modules from: "
                f"{finetune_checkpoint}"
            )
            load_summary = load_non_wavlm_checkpoint(model, finetune_checkpoint)
            accelerator.print(
                "Selective initialization complete: "
                f"loaded {load_summary['loaded_non_wavlm_tensors']} non-WavLM tensors; "
                f"preserved {load_summary['preserved_wavlm_tensors']} stage-2 WavLM tensors; "
                f"ignored {load_summary['ignored_source_wavlm_tensors']} source WavLM tensors."
            )

    optimizer_small = instantiate(
        config["optimizer_small"]["path"],
        args={"params": model.wavlm_model.parameters()}
        | config["optimizer_small"]["args"]
        | {"lr": config["optimizer_small"]["args"]["lr"]},
    )
    optimizer_big = instantiate(
        config["optimizer_big"]["path"],
        args={"params": model.non_wavlm_parameters()}
        | config["optimizer_big"]["args"]
        | {"lr": config["optimizer_big"]["args"]["lr"]},
    )

    (model, optimizer_small, optimizer_big) = accelerator.prepare(
        model, optimizer_small, optimizer_big
    )

    # Pass model receptive-field information to the datasets.
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step

    collate_fn_partial = partial(
        _collate_fn,
        max_speakers_per_chunk=config["model"]["args"]["max_speakers_per_chunk"],
    )

    if "train" in args.mode:
        train_dataset = instantiate(
            config["train_dataset"]["path"], args=train_dataset_config
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=collate_fn_partial,
            shuffle=True,
            **config["train_dataset"]["dataloader"],
        )
        train_dataloader = accelerator.prepare(train_dataloader)

    if "train" in args.mode or "validate" in args.mode:
        validate_dataset = instantiate(
            config["validate_dataset"]["path"], args=validate_dataset_config
        )
        validate_dataloader = DataLoader(
            dataset=validate_dataset,
            collate_fn=collate_fn_partial,
            shuffle=False,
            **config["validate_dataset"]["dataloader"],
        )
        validate_dataloader = accelerator.prepare(validate_dataloader)

    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        accelerator=accelerator,
        config=config,
        resume=resume,
        model=model,
        optimizer_small=optimizer_small,
        optimizer_big=optimizer_big,
    )

    for flag in args.mode:
        if flag == "train":
            trainer.train(train_dataloader, validate_dataloader)
        elif flag == "validate":
            trainer.validate(validate_dataloader)
        else:
            raise ValueError(f"Unknown mode: {flag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-ZEN based EEND framework")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-M",
        "--mode",
        nargs="+",
        type=str,
        default=["train"],
        choices=["train", "validate"],
        help="Mode of the experiment.",
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "-FT",
        "--finetune",
        action="store_true",
        help="Initialize non-WavLM segmentation modules from a checkpoint.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Full segmentation checkpoint used for non-WavLM initialization.",
    )
    parser.add_argument(
        "--pruned_ckpt_path",
        required=True,
        type=str,
        help="Stage-2 pruned WavLM checkpoint (pytorch_model.bin).",
    )

    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().resolve()
    config = toml.load(config_path.as_posix())

    config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()

    pruned_ckpt_path = Path(args.pruned_ckpt_path).expanduser().resolve()
    if not pruned_ckpt_path.is_file():
        parser.error(f"Stage-2 pruned checkpoint file not found: {pruned_ckpt_path}")
    config["model"]["args"]["wavlm_src"] = pruned_ckpt_path.as_posix()

    if args.finetune:
        if not args.ckpt_path:
            parser.error("--ckpt_path is required with --finetune")

        ckpt_path = Path(args.ckpt_path).expanduser().resolve()
        if not ckpt_path.is_file():
            parser.error(f"Fine-tune checkpoint file not found: {ckpt_path}")

        config.setdefault("finetune", {}).update(
            finetune=True,
            ckpt_dir=ckpt_path.as_posix(),
            ckpt_path=ckpt_path.as_posix(),
        )
    elif args.ckpt_path:
        parser.error("--ckpt_path requires --finetune")

    run(config, args.resume)
