# Validation results

This is a historical validation record for the artifact identified below.
Export and GPU parity were not rerun when preparing `split_train_stages`.
The generated ONNX file and detailed JSON reports are external artifacts and
are excluded from Git. Reproduce them using the commands in `README.md` and
record the resolved model revision and checksum for your new run.

Exported model: `BUT-FIT/diarizen-wavlm-large-s80-md`

Resolved Hugging Face revision: `a9b1b0e7974d96dcfd63af417e9da7ad8714040f`

Artifact: `diarizen-wavlm-large-s80-md-segmentation.onnx`

- SHA-256: `66e2d836af5f29220f8be379853289922efe2c799648810955ca72f736d8a205`
- Size: 280,453,353 bytes
- ONNX opset: 17
- Input: `float32 waveform[batch, 1, 256000]`
- Output: `float32 powerset_scores[batch, 799, 11]`
- Dynamic axis: batch only

The graph passed `onnx.checker.check_model(..., full_check=True)`.

## PyTorch parity

The validator loaded a fresh, unmodified PyTorch checkpoint and compared it
against ONNX Runtime using `rtol=1e-3`, `atol=1e-4`.

Test inputs:

1. 16 seconds of silence, batch size 1.
2. A deterministic batch of sine mixture, seeded Gaussian noise, and impulse,
   batch size 3. This also validates the dynamic batch axis.
3. The first 16 seconds of `example/EN2002a_30s.wav`.
4. The exact final pipeline-style window from the example recording: start at
   sample 230400, use 249600 real samples, then append 6400 zeros.

| Runtime | Passed | Worst absolute error | Largest mean absolute error | Frame argmax agreement |
|---|---:|---:|---:|---:|
| ONNX Runtime CUDA | yes | 3.147125244140625e-05 | 4.460944182937965e-06 | 100% for every case |
| ONNX Runtime CPU | yes | 5.779266357421875e-04 | 1.1052738409489393e-04 | 100% for every case |

CUDA profiling confirmed graph execution on `CUDAExecutionProvider` (5304
profiled node events; 1432 shape/control events ran on CPU). CPU validation
used only `CPUExecutionProvider`.

Detailed machine-readable reports:

- `diarizen-wavlm-large-s80-md-segmentation.validation-cuda.json`
- `diarizen-wavlm-large-s80-md-segmentation.validation-cpu.json`
