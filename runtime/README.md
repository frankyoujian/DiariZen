# DiariZen segmentation ONNX runtime

For the full training/evaluation workflow, see the
[restart guide](../docs/SPLIT_TRAIN_STAGES.md). These tools export a released
Hugging Face model; they do not load a locally trained stage-3 checkpoint.
Generated graphs, caches, metadata, and validation JSONs are excluded from Git.
`VALIDATION.md` describes historical results. Downloads use the model's default
revision; record its resolved revision and your artifact checksum when rerunning.

This directory exports and validates the neural segmentation stage of
`BUT-FIT/diarizen-wavlm-large-s80-md`.

The ONNX graph does **not** include the rest of the diarization pipeline:
sliding-window orchestration, median filtering, speaker counting, WeSpeaker
embeddings, VBx/AHC clustering, reconstruction, or RTTM generation.

## Files

- `export_segmentation_onnx.py`: downloads/loads the released checkpoint,
  exports the fixed 16-second segmentation graph, and runs `onnx.checker`.
- `validate_segmentation_onnx.py`: compares fresh PyTorch and ONNX Runtime
  outputs on silence, a synthetic batch, and two chunks from the repository's
  example recording. It also confirms which ORT execution provider ran nodes.
- `run_export_and_validate.sh`: convenience wrapper.
- `requirements-onnx.txt`: exporter/runtime packages used for this artifact.

Generated files:

- `diarizen-wavlm-large-s80-md-segmentation.onnx`
- `diarizen-wavlm-large-s80-md-segmentation.metadata.json`
- `diarizen-wavlm-large-s80-md-segmentation.validation.json`

## Run

From the repository root:

```bash
.venv/bin/python -m pip install -r runtime/requirements-onnx.txt
runtime/run_export_and_validate.sh
```

Or run the phases separately:

```bash
.venv/bin/python runtime/export_segmentation_onnx.py
.venv/bin/python runtime/validate_segmentation_onnx.py --provider cuda
```

The default graph accepts `float32` waveform chunks shaped
`[batch, 1, 256000]` (16 seconds at 16 kHz). Batch is dynamic; channel and
sample axes are fixed. The original released model produces 799 frame-level
powerset outputs with 11 classes.

Validation exits non-zero unless every test satisfies `numpy.allclose` with
`rtol=1e-3`, `atol=1e-4`, frame-level argmax agreement is at least 99.9%, and
the requested ONNX Runtime provider executed graph nodes. Thresholds are
configurable through command-line options.

The convenience wrapper forwards arguments only to the exporter. When changing
the model, cache, or output location, invoke both phases separately and supply
matching `--model-id`, `--cache-dir`, and validator `--onnx` options. Keep the
dynamic batch axis enabled: the validator tests batches of one and three.
The validator defaults to `--provider auto` and can use a visible GPU. Use
`--provider cpu` when GPU resources are occupied, and separate `--report`
filenames to retain CPU and CUDA results. See `--help` on each Python script.
