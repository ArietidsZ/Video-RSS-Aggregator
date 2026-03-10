# Video RSS Aggregator (Qwen 3.5 Vision, 4-bit)

This project has been rebuilt around Qwen 3.5 multimodal models and a strict local VRAM budget.

- Inference backend: Ollama (Windows-native, no WSL required)
- Default models: Qwen 3.5 4-bit (`q4_K_M`) tiers
- Storage: SQLite (`.data/vra.db`)
- API: FastAPI

## Architecture

- `video_rss_aggregator/` contains the current runtime architecture.
- `video_rss_aggregator/bootstrap.py` composes the application runtime and use cases.
- `video_rss_aggregator/application/` holds use-case orchestration and ports.
- `video_rss_aggregator/domain/` defines the core models and outcome types.
- `video_rss_aggregator/infrastructure/` contains SQLite, RSS, media, summarization, and runtime adapters.
- `video_rss_aggregator/gui.py` plus packaged `video_rss_aggregator/templates/` and `video_rss_aggregator/static/` drive the setup studio UI.
- Root modules such as `adapter_api.py`, `adapter_rss.py`, `adapter_storage.py`, and `cli.py` remain as compatibility and entry-point surfaces around the packaged runtime.

## Design Goals

- Use Qwen 3.5 vision-capable small models for summarization quality.
- Keep total app VRAM use within `8GB` by default.
- Prefer 4-bit model variants for quality/efficiency balance.
- Keep setup simple for Windows users (no WSL).
- Use scene-aware frame extraction with timeline coverage for better visual context.

## Requirements

- Python 3.11+
- Windows 10/11
- NVIDIA GPU with at least 8GB VRAM
- Ollama installed on Windows: https://ollama.com/download/windows
- `ffmpeg` and `ffprobe` on `PATH`

## Quick Start (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Run bootstrap (auto-pulls configured models if missing):

```bash
python -m vra bootstrap
```

Start server:

```bash
python -m vra serve --bind 127.0.0.1:8080
```

Then open `http://127.0.0.1:8080/` for the guided installation + configuration GUI.
The setup page includes one-click diagnostics for Python, FFmpeg/FFprobe, yt-dlp, and Ollama reachability, plus a packaged HTML/CSS/JS studio for generating the full `.env` block.

## 4-bit Model Defaults

Default model priority:

1. `qwen3.5:4b-q4_K_M`
2. `qwen3.5:2b-q4_K_M`
3. `qwen3.5:0.8b-q8_0` (safety floor when smaller than 2B is needed)

Each processing job selects one model up front based on configured VRAM budget,
current runtime VRAM usage, and workload size (transcript + frames).
The selected model is pinned for that job; there is no mid-processing model fallback.

## Video Processing Intelligence

- Scene-aware frame candidate extraction (`ffmpeg` scene score) to catch shot changes.
- Uniform timeline sampling fallback/fill to keep temporal coverage when scene cuts are sparse.
- Deduplication by frame content hash before final frame set is sent to the model.
- Model preselection per job using VRAM headroom and estimated per-request overhead.
- SQLite runs in WAL mode with tuned pragmas for better concurrent read/write stability.

## Runtime Commands

```bash
python -m vra bootstrap
python -m vra status
python -m vra verify --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
python -m vra benchmark --source "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

`benchmark` compares `scene_aware` vs `uniform_only` extraction on the same source,
reports frame uniqueness metrics, and (by default) runs both through summarization to
show latency and output-shape differences.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BIND_ADDRESS` | `127.0.0.1:8080` | API bind address |
| `API_KEY` | *(none)* | Optional bearer/API-key auth |
| `VRA_STORAGE_DIR` | `.data` | Download/frame/subtitle storage |
| `VRA_DATABASE_PATH` | `.data/vra.db` | SQLite database path |
| `VRA_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `VRA_MODEL_PRIMARY` | `qwen3.5:4b-q4_K_M` | First-choice model |
| `VRA_MODEL_FALLBACK` | `qwen3.5:2b-q4_K_M` | Second-priority model |
| `VRA_MODEL_MIN` | `qwen3.5:0.8b-q8_0` | Lowest-priority model |
| `VRA_AUTO_PULL_MODELS` | `true` | Pull missing models automatically |
| `VRA_VRAM_BUDGET_MB` | `8192` | Max VRAM budget across the app |
| `VRA_MODEL_SIZE_BUDGET_RATIO` | `0.75` | Share of budget for base model weight |
| `VRA_MODEL_SELECTION_RESERVE_MB` | `768` | VRAM safety reserve kept free during model selection |
| `VRA_CONTEXT_TOKENS` | `3072` | Context window per request |
| `VRA_MAX_OUTPUT_TOKENS` | `768` | Summary output cap |
| `VRA_MAX_FRAMES` | `5` | Max sampled frames per source |
| `VRA_FRAME_SCENE_DETECTION` | `true` | Enable scene-aware frame selection |
| `VRA_FRAME_SCENE_THRESHOLD` | `0.28` | Scene change sensitivity (`ffmpeg` scene score threshold) |
| `VRA_FRAME_SCENE_MIN_FRAMES` | `2` | Minimum detected scene frames before blending with uniform sampling |
| `VRA_MAX_TRANSCRIPT_CHARS` | `16000` | Subtitle transcript cap |
| `VRA_RSS_TITLE` | `Video RSS Aggregator` | RSS title |
| `VRA_RSS_LINK` | `http://127.0.0.1:8080/rss` | RSS self-link |
| `VRA_RSS_DESCRIPTION` | `Video summaries` | RSS description |

## API

- `GET /` (GUI setup + configuration workspace)
- `GET /health`
- `GET /setup/config`
- `GET /setup/diagnostics`
- `POST /setup/bootstrap`
- `GET /runtime`
- `POST /ingest`
- `POST /process`
- `GET /rss?limit=20`

## Notes

- GUI setup/configuration workspace is available at `/` when the server is running.
- The setup studio assets are served from packaged `/static/setup.css` and `/static/setup.js` resources.
- This version is optimized for local, Windows-native operation first.
