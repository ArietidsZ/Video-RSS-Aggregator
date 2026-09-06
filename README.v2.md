# Video RSS Aggregator (v2)

Local-first video RSS pipeline: poll feeds → acquire video (yt-dlp) →
transcript (subtitles, Qwen3-ASR) → scene-aware frames (single ffmpeg
decode pass) → summary by a local Qwen 3.5 vision model → publish as RSS.
One durable SQLite job queue drives everything.

**No fallbacks, one efficient stack.** Exactly one inference path
(OpenAI-compatible server: vLLM / SGLang / oMLX / llama-server / frontier
APIs) and one ASR path (`/v1/audio/transcriptions` hosting Qwen3-ASR).
All defaults are quantized — W4 VLM, q8 KV cache, Q8_0 ASR — because
low-compute speed is a product requirement, and 4-bit ≈ 2× decode
throughput on bandwidth-bound silicon.

## Quick start

```bash
uv sync --extra dev        # or: pip install -e .
vra engine                 # recommended per-platform launch commands + reachability check
vra serve --bind 127.0.0.1:8080
```

Setup studio: `http://127.0.0.1:8080/` (diagnostics + config generator).

```bash
vra ingest https://example.com/channel/videos.xml   # poll a feed once
vra status                                          # counts + jobs
curl -s localhost:8080/rss?limit=20                 # consume the output
```

## Recommended engines (checkpoints are the quantization policy)

| platform | VLM (port 8000) | ASR (port 8081) |
|---|---|---|
| Apple Silicon | oMLX + `lmstudio-community/Qwen3.5-4B-MLX-4bit` + native MTP | `llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF` (Q8_0) |
| Linux/WSL2 + NVIDIA | vLLM + Qwen3.5-4B **AWQ-4bit** + `--kv-cache-dtype fp8` | Qwen3-ASR toolkit on vLLM, or llama-server GGUF |
| CPU-only / generic | `llama-server -hf unsloth/Qwen3.5-4B-GGUF` Q4_K_M + `--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn` | llama-server ASR instance |

Frontier APIs work through the same wire protocol: point
`VRA_ENGINE_BASE_URL` + `VRA_ENGINE_API_KEY` at any provider.

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
                        Qwen3-ASR (/v1/audio/      ffmpeg frame pass
                        transcriptions)            scene+uniform, 1 decode
                                    │                     │
                                    └───────────┬─────────┘
                                                ▼
                        VLM summarize (json_schema strict, map-reduce for
                        long transcripts, quality gate + 1 think-retry)
                                                ▼
                                 SQLite ◄── RSS 2.0 + media:thumbnail
```

- **Durable queue**: `jobs` table; retries with exponential backoff; stale
  running jobs requeued on boot; `VRA_CPU_WORKERS` concurrent process
  workers; inference concurrency is the engine's job (continuous batching).
- **Context math**: transcript tokens (CJK-aware) + image tokens (from
  JPEG SOF dimensions) + output cap + margin against `VRA_MODEL_CTX`;
  long transcripts map-reduce instead of being truncated silently.
- **Frames**: one ffmpeg decode emits scene-hit candidates (real
  per-frame scene scores via `metadata=print`) and a uniform grid; dHash
  (9×8 grayscale, hamming ≤ 6) removes near-duplicates; final set is
  timeline-spread and capped at `VRA_MAX_FRAMES`.
- **Accuracy is benchmark-backed, not aspirational** — `docs/v2/RESEARCH.md`:
  Qwen3-ASR-1.7B matches frontier ASR APIs on zh (WER 4.97 vs
  GPT-4o-Transcribe 15.30 on WenetSpeech); qwen3.5-4b beats GPT-5-Nano /
  Gemini-2.5-Flash-Lite on VideoMME by 9–12 pts. Peak resident ≈ 6.7 GB
  on an 8 GB Mac with the default quants.

## Configuration

All env vars (prefix `VRA_`, plus `BIND_ADDRESS`/`API_KEY`) are listed in
`docs/v2/SPEC.md` and rendered by the setup studio. Key ones:

| var | default | meaning |
|---|---|---|
| `VRA_ENGINE_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible server |
| `VRA_MODEL` | `qwen3.5-4b` | single served model, verified via /v1/models |
| `VRA_MODEL_CTX` | `32768` | server max_model_len (drives map-reduce) |
| `VRA_ASR_BASE_URL` | engine base | ASR server (typical: 8081) |
| `VRA_ASR_MODEL` | `Qwen3-ASR-1.7B` | Q8_0 GGUF default posture |
| `VRA_MAX_FRAMES` | `8` | frames per video sent to the VLM |
| `VRA_ASR` | `auto` | `off` disables ASR entirely |
| `VRA_POLL_MINUTES` | `15` | default feed poll interval |
| `VRA_CPU_WORKERS` | `2` | concurrent download/decode pipelines |
| `VRA_THINK` | `off` | reasoning_effort hint for the final pass |
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
uv run pytest tests/v2 -q     # hermetic: no network, no live engine
uv run ruff check vra tests
```

Architecture and frozen contracts: `docs/v2/SPEC.md`. Per-stage evidence
(WER/VideoMME tables, quantization policy, engine matrix):
`docs/v2/RESEARCH.md`. Deliberately rejected alternatives with evidence:
`docs/v2/ABLATED.md`.

## Requirements

- Python 3.11+ (Windows / macOS / Linux), `ffmpeg`+`ffprobe` on PATH
  (Windows winget Gyan.FFmpeg layout auto-detected)
- An OpenAI-compatible engine serving `VRA_MODEL` — see `vra engine`
  for the exact per-platform commands; ASR server optional
  (`VRA_ASR=off` to skip)
