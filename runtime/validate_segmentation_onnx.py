#!/usr/bin/env python3
"""Validate a DiariZen segmentation ONNX model against fresh PyTorch output."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import soundfile as sf
import torch  # Import before onnxruntime so its CUDA/cuDNN libraries are visible.
import torchaudio
import toml
from huggingface_hub import hf_hub_download
from pyannote.audio import Model

import onnxruntime as ort


RUNTIME_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_DIR.parent
DEFAULT_MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md"
DEFAULT_ONNX = RUNTIME_DIR / "diarizen-wavlm-large-s80-md-segmentation.onnx"
DEFAULT_CACHE = RUNTIME_DIR / "cache"
DEFAULT_AUDIO = REPO_ROOT / "example" / "EN2002a_30s.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--example-audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--provider",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.999)
    return parser.parse_args()


def choose_provider(requested: str) -> str:
    available = ort.get_available_providers()
    cuda_ready = torch.cuda.is_available() and "CUDAExecutionProvider" in available
    if requested == "auto":
        return "cuda" if cuda_ready else "cpu"
    if requested == "cuda" and not cuda_ready:
        raise RuntimeError(
            "CUDA validation requested, but PyTorch/ONNX Runtime CUDA is unavailable. "
            f"ORT providers: {available}"
        )
    return requested


def create_session(
    onnx_path: Path,
    provider: str,
    cuda_device: int,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = "/tmp/diarizen_ort_profile"
    if provider == "cuda":
        providers: list[Any] = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": str(cuda_device),
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=providers,
    )


def static_dimension(value: Any, name: str) -> int:
    if isinstance(value, int) and value > 0:
        return value
    raise RuntimeError(f"Expected static {name}, got {value!r}")


def fit_chunk(waveform: np.ndarray, num_samples: int, from_end: bool = False) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size >= num_samples:
        if from_end:
            waveform = waveform[-num_samples:]
        else:
            waveform = waveform[:num_samples]
    else:
        waveform = np.pad(waveform, (0, num_samples - waveform.size))
    return waveform[None, None, :].astype(np.float32, copy=False)


def load_audio_cases(
    path: Path,
    sample_rate: int,
    num_samples: int,
) -> list[tuple[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Example audio does not exist: {path}")
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio[:, 0]
    if source_rate != sample_rate:
        tensor = torch.from_numpy(mono).unsqueeze(0)
        mono = (
            torchaudio.functional.resample(tensor, source_rate, sample_rate)
            .squeeze(0)
            .numpy()
        )
    return [
        ("example_audio_first", fit_chunk(mono, num_samples, from_end=False)),
        (
            "example_audio_pipeline_tail",
            fit_chunk(
                mono[
                    0
                    if mono.size <= num_samples
                    else (
                        int((mono.size - num_samples) // max(1, round(0.1 * num_samples)))
                        + 1
                    )
                    * max(1, round(0.1 * num_samples)) :
                ],
                num_samples,
            ),
        ),
    ]


def build_cases(
    num_samples: int,
    sample_rate: int,
    seed: int,
    audio_path: Path,
) -> list[tuple[str, np.ndarray]]:
    time_axis = np.arange(num_samples, dtype=np.float64) / sample_rate
    sine = (
        0.08 * np.sin(2.0 * math.pi * 220.0 * time_axis)
        + 0.03 * np.sin(2.0 * math.pi * 440.0 * time_axis + 0.4)
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = (0.015 * rng.standard_normal(num_samples)).astype(np.float32)
    impulse = np.zeros(num_samples, dtype=np.float32)
    impulse[num_samples // 3] = 0.5
    synthetic_batch = np.stack([sine, noise, impulse], axis=0)[:, None, :]

    cases: list[tuple[str, np.ndarray]] = [
        ("silence", np.zeros((1, 1, num_samples), dtype=np.float32)),
        ("synthetic_batch", synthetic_batch),
    ]
    cases.extend(load_audio_cases(audio_path, sample_rate, num_samples))
    return cases


def compare_outputs(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    atol: float,
    rtol: float,
    min_argmax_agreement: float,
) -> dict[str, Any]:
    if pytorch_output.shape != onnx_output.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "pytorch_shape": list(pytorch_output.shape),
            "onnx_shape": list(onnx_output.shape),
        }
    finite = bool(np.isfinite(pytorch_output).all() and np.isfinite(onnx_output).all())
    difference = np.abs(pytorch_output - onnx_output)
    denominator = np.maximum(np.abs(pytorch_output), 1e-6)
    relative = difference / denominator
    argmax_agreement = float(
        np.mean(np.argmax(pytorch_output, axis=-1) == np.argmax(onnx_output, axis=-1))
    )
    allclose = bool(
        np.allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol)
    )
    passed = finite and allclose and argmax_agreement >= min_argmax_agreement
    return {
        "passed": passed,
        "finite": finite,
        "allclose": allclose,
        "pytorch_shape": list(pytorch_output.shape),
        "onnx_shape": list(onnx_output.shape),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_relative_error": float(relative.max(initial=0.0)),
        "p99_relative_error": float(np.percentile(relative, 99.0)),
        "argmax_agreement": argmax_agreement,
    }


def read_profile(session: ort.InferenceSession) -> dict[str, int]:
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text())
    providers = Counter(
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    )
    profile_path.unlink(missing_ok=True)
    return dict(sorted(providers.items()))


def main() -> None:
    args = parse_args()
    started = time.time()
    onnx_path = args.onnx.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    audio_path = args.example_audio.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else onnx_path.with_suffix(".validation.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

    print("Checking the ONNX graph ...", flush=True)
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=True)

    provider = choose_provider(args.provider)
    if provider == "cuda":
        torch.cuda.set_device(args.cuda_device)
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    print(f"Creating ONNX Runtime session on {provider} ...", flush=True)
    session = create_session(onnx_path, provider, args.cuda_device)
    ort_input = session.get_inputs()[0]
    ort_output = session.get_outputs()[0]
    input_shape = ort_input.shape
    if len(input_shape) != 3:
        raise RuntimeError(f"Expected rank-3 input, got {input_shape}")
    channels = static_dimension(input_shape[1], "channel dimension")
    num_samples = static_dimension(input_shape[2], "sample dimension")
    if channels != 1:
        raise RuntimeError(f"Expected a mono export, got {channels} channels")

    config_path = Path(
        hf_hub_download(
            repo_id=args.model_id,
            filename="config.toml",
            cache_dir=str(cache_dir),
        )
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=args.model_id,
            filename="pytorch_model.bin",
            cache_dir=str(cache_dir),
        )
    )
    config = toml.load(config_path)
    print(f"Loading fresh PyTorch model {args.model_id} on {device} ...", flush=True)
    pytorch_model = Model.from_pretrained(checkpoint_path, config=config)
    if pytorch_model is None:
        raise RuntimeError(f"Could not load model {args.model_id!r}.")
    pytorch_model.eval().to(device)
    sample_rate = int(getattr(pytorch_model, "sample_rate", 16000))

    test_cases = build_cases(num_samples, sample_rate, args.seed, audio_path)
    case_results: list[dict[str, Any]] = []
    torch.set_grad_enabled(False)
    for name, waveform in test_cases:
        print(f"Comparing case {name!r}, input {tuple(waveform.shape)} ...", flush=True)
        torch_input = torch.from_numpy(waveform).to(device)
        if provider == "cuda":
            torch.cuda.synchronize(device)
        torch_started = time.perf_counter()
        with torch.inference_mode():
            pytorch_output = pytorch_model(torch_input).detach().cpu().numpy()
        if provider == "cuda":
            torch.cuda.synchronize(device)
        torch_seconds = time.perf_counter() - torch_started

        ort_started = time.perf_counter()
        onnx_output = session.run(
            [ort_output.name],
            {ort_input.name: waveform},
        )[0]
        ort_seconds = time.perf_counter() - ort_started

        metrics = compare_outputs(
            pytorch_output,
            onnx_output,
            atol=args.atol,
            rtol=args.rtol,
            min_argmax_agreement=args.min_argmax_agreement,
        )
        metrics.update(
            {
                "name": name,
                "input_shape": list(waveform.shape),
                "pytorch_seconds": torch_seconds,
                "onnxruntime_seconds": ort_seconds,
            }
        )
        case_results.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    provider_node_counts = read_profile(session)
    expected_provider = (
        "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    )
    provider_executed = provider_node_counts.get(expected_provider, 0) > 0
    passed = all(case["passed"] for case in case_results) and provider_executed
    report = {
        "passed": passed,
        "onnx_path": str(onnx_path),
        "model_id": args.model_id,
        "example_audio": str(audio_path),
        "provider_requested": args.provider,
        "provider_used": provider,
        "session_providers": session.get_providers(),
        "provider_node_counts": provider_node_counts,
        "expected_provider_executed": provider_executed,
        "input_name": ort_input.name,
        "input_shape": input_shape,
        "output_name": ort_output.name,
        "output_shape": ort_output.shape,
        "sample_rate": sample_rate,
        "thresholds": {
            "atol": args.atol,
            "rtol": args.rtol,
            "min_argmax_agreement": args.min_argmax_agreement,
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "cases": case_results,
        "elapsed_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Validation report: {report_path}", flush=True)
    if not passed:
        raise SystemExit("ONNX parity validation failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
