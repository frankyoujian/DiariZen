#!/usr/bin/env python3
"""Export a released DiariZen segmentation network to ONNX.

This intentionally exports only the neural segmentation model. Sliding-window
chunking, speaker embeddings, clustering, and diarization reconstruction remain
outside the ONNX graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import onnx
import toml
import torch
from huggingface_hub import hf_hub_download
from pyannote.audio import Model
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


RUNTIME_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_DIR.parent
DEFAULT_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md"
DEFAULT_OUTPUT = RUNTIME_DIR / "diarizen-wavlm-large-s80-md-segmentation.onnx"
DEFAULT_CACHE = RUNTIME_DIR / "cache"


class SegmentationExportWrapper(nn.Module):
    """Expose a single tensor input and output for ONNX export."""

    def __init__(
        self,
        model: nn.Module,
        num_samples: int,
        normalize_waveform: bool,
    ):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.normalize_waveform = normalize_waveform
        if normalize_waveform:
            self.register_buffer("norm_weight", torch.ones(num_samples))
            self.register_buffer("norm_bias", torch.zeros(num_samples))
        else:
            self.register_buffer("norm_weight", None)
            self.register_buffer("norm_bias", None)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.normalize_waveform:
            waveform = F.layer_norm(
                waveform, (self.num_samples,), self.norm_weight, self.norm_bias
            )
        return self.model(waveform)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        help="Override the segment duration from the model's config.toml.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make only the batch axis dynamic. The audio length stays fixed.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def bake_parametrizations(module: nn.Module) -> list[str]:
    """Bake inference-time parametrizations, notably WavLM weight norm."""

    removed: list[str] = []
    for name, child in module.named_modules():
        parametrizations = getattr(child, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name in list(parametrizations.keys()):
            parametrize.remove_parametrizations(
                child, parameter_name, leave_parametrized=True
            )
            removed.append(f"{name or '<root>'}.{parameter_name}")
    return removed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shape(value_info: Any) -> list[int | str | None]:
    result: list[int | str | None] = []
    tensor_type = value_info.type.tensor_type
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    metadata_output = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else output.with_suffix(".metadata.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    print(f"Loading model metadata for {args.model_id} ...", flush=True)
    config_path = Path(
        hf_hub_download(
            repo_id=args.model_id,
            filename="config.toml",
            cache_dir=str(cache_dir),
        )
    )
    config = toml.load(config_path)
    chunk_seconds = args.chunk_seconds
    if chunk_seconds is None:
        chunk_seconds = float(config["inference"]["args"]["seg_duration"])

    print(f"Loading PyTorch checkpoint for {args.model_id} ...", flush=True)
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=args.model_id,
            filename="pytorch_model.bin",
            cache_dir=str(cache_dir),
        )
    )
    model = Model.from_pretrained(checkpoint_path, config=config)
    if model is None:
        raise RuntimeError(f"Could not load model {args.model_id!r}.")
    model.eval().cpu()

    sample_rate = int(getattr(model, "sample_rate", 16000))
    num_samples = int(round(chunk_seconds * sample_rate))
    if num_samples <= 0:
        raise ValueError(f"Invalid input length: {num_samples} samples")

    # GradMultiply changes gradients only. Bypassing it removes a custom
    # autograd.Function from the inference graph without changing outputs.
    wavlm_model = getattr(model, "wavlm_model", None)
    original_feature_grad_mult = None
    original_normalize_waveform = False
    if wavlm_model is not None and hasattr(wavlm_model, "feature_grad_mult"):
        original_feature_grad_mult = float(wavlm_model.feature_grad_mult)
        wavlm_model.feature_grad_mult = 1.0
    if wavlm_model is not None and hasattr(wavlm_model, "normalize_waveform"):
        original_normalize_waveform = bool(wavlm_model.normalize_waveform)
        wavlm_model.normalize_waveform = False

    removed_parametrizations = bake_parametrizations(model)
    wrapper = SegmentationExportWrapper(
        model,
        num_samples=num_samples,
        normalize_waveform=original_normalize_waveform,
    ).eval()
    dummy = torch.randn(
        args.batch_size,
        1,
        num_samples,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        reference = wrapper(dummy)
    if reference.ndim != 3:
        raise RuntimeError(
            f"Expected [batch, frames, classes], got {tuple(reference.shape)}"
        )

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "waveform": {0: "batch"},
            "powerset_scores": {0: "batch"},
        }

    print(
        f"Exporting {tuple(dummy.shape)} -> {tuple(reference.shape)} "
        f"to {output} (opset {args.opset}) ...",
        flush=True,
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["waveform"],
            output_names=["powerset_scores"],
            dynamic_axes=dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )

    print("Running ONNX structural validation ...", flush=True)
    onnx_model = onnx.load(str(output), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.helper.set_model_props(
        onnx_model,
        {
            "source_model": args.model_id,
            "source_repository": "https://github.com/BUTSpeechFIT/DiariZen",
            "scope": "neural segmentation only",
            "sample_rate": str(sample_rate),
            "chunk_seconds": str(chunk_seconds),
            "num_samples": str(num_samples),
            "pytorch_version": torch.__version__,
        },
    )
    onnx.save(onnx_model, str(output))
    onnx.checker.check_model(str(output), full_check=True)

    graph_inputs = {
        value.name: tensor_shape(value) for value in onnx_model.graph.input
    }
    graph_outputs = {
        value.name: tensor_shape(value) for value in onnx_model.graph.output
    }
    opsets = {item.domain or "ai.onnx": item.version for item in onnx_model.opset_import}
    metadata = {
        "artifact": str(output),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "model_id": args.model_id,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "sample_rate": sample_rate,
        "chunk_seconds": chunk_seconds,
        "num_samples": num_samples,
        "example_input_shape": list(dummy.shape),
        "example_output_shape": list(reference.shape),
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "opsets": opsets,
        "dynamic_batch": args.dynamic_batch,
        "feature_grad_mult_before_export": original_feature_grad_mult,
        "normalize_waveform_in_wrapper": original_normalize_waveform,
        "removed_parametrizations": removed_parametrizations,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "elapsed_seconds": time.time() - started,
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Export completed: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
