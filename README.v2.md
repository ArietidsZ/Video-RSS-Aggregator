# Video RSS Aggregator (v2)

Local-first video RSS pipeline: poll feeds → acquire video (yt-dlp) →
transcript (subtitles, Qwen3-ASR chain) → scene-aware frames
(single ffmpeg decode pass) → summary by a local Qwen 3.5 vision model on
Ollama → publish as RSS. One durable SQLite job queue drives everything.

## Quick start

```bash
uv sync --extra dev        # or: pip install -e .
vra bootstrap              # verify Ollama, pull configured models
vra serve --bind 127.0.0.1:8080
```

Setup studio: `http://127.0.0.1:8080/` (diagnostics + config generator).

```bash
vra ingest https://example.com/channel/videos.xml   # poll a feed once
vra status                                          # counts + jobs
curl -s localhost:8080/rss?limit=20                 # consume the output
```

## How it works

```
feeds → [ingest job] → entries ──► videos(pending) ──► [process job]
      scheduler (poll_minutes, jitter)                    │
                                    ┌─────────────────────┤
                                    ▼                     ▼
                             yt-dlp metadata+subs   yt-dlp download
                             (cached, idempotent)    (skipped if cached)
                                    │                     │
                                    ▼                     ▼
                        Qwen3-ASR chain           ffmpeg frame pass
                        (mlx-audio / vLLM /       scene+uniform, 1 decode
                        llama.cpp; faster-whisper)
                                    └───────────┬───────────┘
                                                ▼
                              VLM summarize (schema JSON, VRAM governor,
                              map-reduce for long transcripts)
                                                ▼
                                 SQLite ◄── RSS 2.0 + media:thumbnail
```

- **Durable queue**: `jobs` table; retries with exponential backoff; stale
  running jobs requeued on boot; `VRA_CPU_WORKERS` concurrent process
  workers; GPU stage serialized by a semaphore (`VRA_GPU_WORKERS`) so
  downloads of the next video overlap inference of the current one.
- **VRAM governor**: picks the first model from `VRA_MODELS` whose file
  size (+KV estimate) fits `VRA_VRAM_BUDGET_MB - VRA_RESERVE_MB - current
  VRAM`. MLX variants preferred automatically on Apple Silicon. Model
  pinned with `VRA_KEEP_ALIVE` during bursts, unloaded on idle.
- **Context math**: transcript tokens (CJK-aware) + image tokens (from
  JPEG SOF dimensions) + output cap + margin, rounded to 512. Long
  transcripts map-reduce instead of being truncated silently.
- **Frames**: one ffmpeg decode emits scene-hit candidates (with real
  per-frame scene scores from `metadata=print`) and a uniform grid;
  dHash (9×8 grayscale, hamming ≤ 6) removes near-duplicates; final set
  is timeline-spread and capped at `VRA_MAX_FRAMES`.
- **Accuracy guardrails**: structured output via JSON schema; `think`
  escalation retry on low-quality summaries; extractive fallback from the
  transcript; transcript language matching; nothing invented that isn't
  in transcript/frames — by prompt contract and gate, not hope.

## Configuration

All env vars (prefix `VRA_`, plus `BIND_ADDRESS`/`API_KEY`) with defaults
are listed in `docs/v2/SPEC.md` and rendered by the setup studio. Key ones:

| var | default | meaning |
|---|---|---|
| `VRA_MODELS` | `qwen3.5:4b,qwen3.5:2b,qwen3.5:0.8b` | priority order, first fit wins |
| `VRA_VRAM_BUDGET_MB` | `8192` | total model+KV budget |
| `VRA_MAX_FRAMES` | `8` | frames per video sent to the VLM |
| `VRA_ASR` | `auto` | transcribe when no subtitles (Qwen3-ASR chain) |
| `VRA_POLL_MINUTES` | `15` | default feed poll interval |
| `VRA_CPU_WORKERS` / `VRA_GPU_WORKERS` | `2` / `1` | download/decode vs inference concurrency |
| `VRA_THINK` | `off` | reasoning level for the final summary pass |
| `API_KEY` | *(none)* | enables bearer auth on mutating endpoints |

## API

`GET /health` · `GET|POST|DELETE /feeds` · `POST /ingest` · `POST
/process` · `GET /jobs[/{id}]` · `GET /videos[/{id}]` · `GET /rss` ·
`GET /metrics` · `GET /setup/config|diagnostics` · `POST
/setup/bootstrap` · static frames at `/media/{video_id}/{idx}.jpg`.

RSS output includes `media:thumbnail` (first frame) and a `<ul>` of key
points in the description; validates in standard readers.

## Development

```bash
uv sync --extra dev
uv run pytest tests/v2 -q     # hermetic: no network, no live Ollama
uv run ruff check vra tests
```

Architecture and frozen contracts: `docs/v2/SPEC.md`. Deliberately
rejected alternatives with evidence: `docs/v2/ABLATED.md`.

## Requirements

- Python 3.11+ (Windows / macOS / Linux), `ffmpeg`+`ffprobe` on PATH
  (Windows winget Gyan.FFmpeg layout auto-detected)
- Ollama running locally (`VRA_OLLAMA_BASE_URL`), models pulled by `vra bootstrap`
- Optional ASR runtimes (subtitles need nothing): `pip install mlx-audio`
  (Apple Silicon) or `pip install .[asr]` (faster-whisper CPU fallback),
  or point `VRA_ENGINE_BASE_URL` at a Qwen3-ASR vLLM/llama-server.
