#!/usr/bin/env python3
"""Convert Microsoft's original WavLM checkpoint to DiariZen's current schema."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch

from diarizen.models.module.wav2vec2.model import wavlm_model
from diarizen.models.module.wavlm_config import get_config


def map_key(key: str) -> str | None:
    if key == "mask_emb":
        return None

    match = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.weight", key)
    if match:
        return f"feature_extractor.conv_layers.{match.group(1)}.conv.weight"

    match = re.fullmatch(r"feature_extractor\.conv_layers\.0\.2\.(weight|bias)", key)
    if match:
        return f"feature_extractor.conv_layers.0.layer_norm.{match.group(1)}"

    if key.startswith("post_extract_proj."):
        return key.replace(
            "post_extract_proj.", "encoder.feature_projection.projection.", 1
        )
    if key.startswith("layer_norm."):
        return key.replace(
            "layer_norm.", "encoder.feature_projection.layer_norm.", 1
        )
    if key.startswith("encoder.pos_conv.0."):
        suffix = key.removeprefix("encoder.pos_conv.0.")
        suffix = {
            "weight_g": "parametrizations.weight.original0",
            "weight_v": "parametrizations.weight.original1",
        }.get(suffix, suffix)
        return f"encoder.transformer.pos_conv_embed.conv.{suffix}"
    if key.startswith("encoder.layer_norm."):
        return key.replace("encoder.layer_norm.", "encoder.transformer.layer_norm.", 1)

    match = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key)
    if not match:
        raise KeyError(f"No conversion rule for original key: {key}")

    layer, suffix = match.groups()
    replacements = (
        ("self_attn.relative_attention_bias.", "attention.rel_attn_embed."),
        ("self_attn.grep_a", "attention.gru_rel_pos_const"),
        ("self_attn.grep_linear.", "attention.gru_rel_pos_linear."),
        ("self_attn.", "attention."),
        ("self_attn_layer_norm.", "layer_norm."),
        ("fc1.", "feed_forward.intermediate_dense."),
        ("fc2.", "feed_forward.output_dense."),
        ("final_layer_norm.", "final_layer_norm."),
    )
    for original_prefix, target_prefix in replacements:
        if suffix == original_prefix or suffix.startswith(original_prefix):
            suffix = target_prefix + suffix[len(original_prefix) :]
            break
    else:
        raise KeyError(f"No conversion rule for encoder-layer key: {key}")

    return f"encoder.transformer.layers.{layer}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    original_checkpoint = torch.load(source, map_location="cpu")
    if set(original_checkpoint) != {"cfg", "model"}:
        raise ValueError(
            f"Expected Microsoft's cfg/model checkpoint, got: "
            f"{sorted(original_checkpoint)}"
        )

    config = get_config("wavlm_base")
    target_model = wavlm_model(**config)
    target_template = target_model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    ignored: list[str] = []

    for original_key, tensor in original_checkpoint["model"].items():
        target_key = map_key(original_key)
        if target_key is None:
            ignored.append(original_key)
            continue
        if target_key not in target_template:
            raise KeyError(f"Mapped key does not exist in target model: {target_key}")
        if target_template[target_key].shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {original_key} -> {target_key}: "
                f"{tuple(tensor.shape)} != {tuple(target_template[target_key].shape)}"
            )
        if target_key in converted:
            raise KeyError(f"Duplicate mapped target key: {target_key}")
        converted[target_key] = tensor

    default_initialized = sorted(set(target_template) - set(converted))
    if default_initialized != ["feature_extractor.dummy_weight"]:
        raise ValueError(
            f"Unexpected target parameters without source weights: {default_initialized}"
        )
    converted["feature_extractor.dummy_weight"] = target_template[
        "feature_extractor.dummy_weight"
    ]

    incompatible = target_model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict load failed: {incompatible}")

    converted_checkpoint = {
        "config": config,
        "state_dict": converted,
        "conversion": {
            "source": str(source),
            "source_schema": ["cfg", "model"],
            "ignored_source_keys": ignored,
            "default_initialized_target_keys": default_initialized,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(converted_checkpoint, temporary)
        verified = torch.load(temporary, map_location="cpu")
        verification_model = wavlm_model(**verified["config"])
        verification_model.load_state_dict(verified["state_dict"], strict=True)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        {
            "source": str(source),
            "output": str(output),
            "output_bytes": output.stat().st_size,
            "converted_tensors": len(converted),
            "ignored_source_keys": ignored,
            "default_initialized_target_keys": default_initialized,
        }
    )


if __name__ == "__main__":
    main()
