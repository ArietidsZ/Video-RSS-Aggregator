# Video RSS Aggregator

Intelligent video summarization and RSS feed generation powered by Qwen3 models on NVIDIA CUDA.

- **ASR**: Qwen/Qwen3-ASR-1.7B (via `qwen-asr`)
- **Summarization**: Qwen/Qwen3-8B-AWQ (via vLLM)
- **Storage**: PostgreSQL
- **API**: FastAPI

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (Windows / Linux)
- PostgreSQL 15+
- ffmpeg on PATH

## Quick Start

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux

# Install
pip install -e .

# Configure
set DATABASE_URL=postgresql://user:pass@localhost:5432/video_rss

# Run
python -m vra serve --bind 0.0.0.0:8080
```

Models are downloaded from Hugging Face automatically on first run.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string |
| `BIND_ADDRESS` | `0.0.0.0:8080` | HTTP server bind address |
| `API_KEY` | *(none)* | Optional bearer token for auth |
| `VRA_STORAGE_DIR` | `.data` | Local storage for downloads and audio |
| `VRA_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ASR model name or path |
| `VRA_LLM_MODEL` | `Qwen/Qwen3-8B-AWQ` | Summarization model name or path |
| `VRA_GPU_MEMORY_UTILIZATION` | `0.8` | vLLM GPU memory fraction |
| `VRA_ASR_DEVICE` | `cuda:0` | PyTorch device for ASR |
| `VRA_ASR_MAX_TOKENS` | `4096` | Max tokens for ASR output |
| `VRA_LLM_MAX_TOKENS` | `2048` | Max tokens for summarization |
| `VRA_RSS_TITLE` | `Video RSS Aggregator` | RSS feed title |
| `VRA_RSS_LINK` | `http://localhost:8080/rss` | RSS feed self-link |
| `VRA_RSS_DESCRIPTION` | `Video summaries` | RSS feed description |

## API

### `GET /health`
Returns health status.

### `POST /ingest`
Ingest an RSS/Atom feed. Body: `{"feed_url": "...", "process": true, "max_items": 5}`

### `POST /process`
Process a single video/audio source. Body: `{"source_url": "...", "title": "..."}`

### `GET /rss?limit=20`
Returns summarized content as RSS 2.0 XML.

## Verification

```bash
python -m vra verify --feed-url "https://example.com/feed.xml" --source "/path/to/audio.wav"
```
