# Video RSS Aggregator (Native)

Cross-platform video RSS ingestion, transcription, and summarization with native acceleration.

## Goals
- Native acceleration on macOS (MPS + CoreML/NPU), Windows (CUDA/ROCm/oneAPI), Linux (CUDA/ROCm/oneAPI).
- Minimal, efficient codebase with a single Rust binary.
- Deterministic storage in PostgreSQL and RSS output.

## Requirements
- Rust 1.80+ (edition 2024)
- PostgreSQL 15+
- ffmpeg on PATH for audio extraction
- Accelerator backend libraries (see below)

## Quick Start
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/video_rss"
export VRA_ACCEL=auto
export VRA_ACCEL_LIB_DIR="/opt/vra-backends"

cargo run --release -- serve --bind 0.0.0.0:8080
```

## Apple Silicon Quick Start (Minimum Steps)
```bash
cargo run --release -- setup
export DATABASE_URL="postgresql://user:pass@localhost:5432/video_rss"
export VRA_TRANSCRIBE_MODEL_PATH="/path/to/whisper.gguf"
export VRA_SUMMARIZE_MODEL_PATH="/path/to/llama.gguf"
cargo run --release -- serve --bind 0.0.0.0:8080
```
The setup command builds MPS (transcription) and CoreML (summarization) backends into `.data/backends`
and becomes the default on macOS, so no additional flags are needed.

## Acceleration Backends
This binary loads platform-specific accelerator plugins via a small C ABI.
Provide the shared library in `VRA_ACCEL_LIB_DIR` or `VRA_ACCEL_LIB_NAME`.

Default library names:
- macOS: `libvra_coreml_backend.dylib`, `libvra_mps_backend.dylib`
- Windows: `vra_cuda_backend.dll`, `vra_rocm_backend.dll`, `vra_oneapi_backend.dll`
- Linux: `libvra_cuda_backend.so`, `libvra_rocm_backend.so`, `libvra_oneapi_backend.so`

Required C symbols:
- `vra_backend_init(config_json: *const c_char) -> i32`
- `vra_backend_transcribe(audio_path: *const c_char, output_json: *mut *const c_char) -> i32`
- `vra_backend_summarize(text: *const c_char, output_json: *mut *const c_char) -> i32`
- `vra_backend_free_string(ptr: *const c_char)`

## Backend Build
The native wrapper lives in `backends/` and links against `whisper.cpp` and `llama.cpp`
that you build with the desired accelerator backend (Metal/MPS, CoreML, CUDA, ROCm, oneAPI).
It expects recent C APIs from both projects; adjust `backends/src/vra_backend.cpp` if you use older versions.

Example build:
```bash
cmake -S backends -B backends/build \
  -DVRA_BACKEND_KIND=cuda \
  -DWHISPER_CPP_INCLUDE_DIR=/opt/whisper.cpp/include \
  -DWHISPER_CPP_LIB_DIR=/opt/whisper.cpp/build \
  -DLLAMA_CPP_INCLUDE_DIR=/opt/llama.cpp/include \
  -DLLAMA_CPP_LIB_DIR=/opt/llama.cpp/build

cmake --build backends/build --config Release
```
The build outputs `vra_<backend>_backend` with the platform-specific prefix/suffix
(e.g., `libvra_cuda_backend.so`, `vra_cuda_backend.dll`, `libvra_mps_backend.dylib`).

## Environment Variables
- `DATABASE_URL` (required)
- `BIND_ADDRESS` (default: `0.0.0.0:8080`)
- `API_KEY` (optional bearer token)
- `VRA_STORAGE_DIR` (default: `.data`)
- `VRA_ACCEL` = `auto|mps|coreml|cuda|rocm|oneapi|cpu`
- `VRA_ALLOW_CPU` = `0|1` (default: `0`)
- `VRA_ACCEL_LIB_DIR` (directory containing backend library)
- `VRA_ACCEL_LIB_NAME` (explicit library file name)
- `VRA_ACCEL_DEVICE` (device selector passed to backend)
- `VRA_TRANSCRIBE_ACCEL`, `VRA_SUMMARIZE_ACCEL` (override backend per purpose)
- `VRA_TRANSCRIBE_LIB_DIR`, `VRA_SUMMARIZE_LIB_DIR` (override library directory per purpose)
- `VRA_TRANSCRIBE_LIB_NAME`, `VRA_SUMMARIZE_LIB_NAME` (override library name per purpose)
- `VRA_TRANSCRIBE_DEVICE`, `VRA_SUMMARIZE_DEVICE` (override device selector per purpose)
- `VRA_WHISPER_REPO`, `VRA_LLAMA_REPO` (override setup git URLs)
- `VRA_WHISPER_REF`, `VRA_LLAMA_REF` (checkout a specific git ref during setup)
- `VRA_WHISPER_CMAKE_ARGS`, `VRA_LLAMA_CMAKE_ARGS`, `VRA_BACKEND_CMAKE_ARGS` (extra CMake flags)
- `VRA_TRANSCRIBE_MODEL_PATH` (required for transcription)
- `VRA_SUMMARIZE_MODEL_PATH` (required for summarization)
- `VRA_VERIFY_FEED_URL`, `VRA_VERIFY_AUDIO_PATH`, `VRA_VERIFY_AUDIO_URL`, `VRA_VERIFY_VIDEO_URL` (verification inputs)

## Verification
Run OS-level verification with real data only:
```bash
cargo run --release -- verify
```
If any required real input is missing, verification fails with a clear error.
