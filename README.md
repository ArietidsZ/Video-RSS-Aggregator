# Video RSS Aggregator (v2)

Local-first video RSS pipeline: poll feeds → acquire video (yt-dlp) →
transcript (subtitles, Qwen3-ASR) → scene-aware frames (single ffmpeg
decode pass) → summary by a local Qwen 3.5 vision model → publish as RSS.
One durable SQLite job queue drives everything.

**No fallbacks, one efficient stack, one posture.** Exactly one inference
path (an OpenAI-compatible server, engine-agnostic by contract) and one
ASR path (`/v1/audio/transcriptions`). One model choice, chosen by
iso-footprint scaling: the **largest model at the most aggressive
verified quantization** that fits the ~4 GB VRAM / ~4 GB RAM envelope —
target **Qwen3.5-9B @ ~2.5 bpw GSQ/RCO-refined weights + quantized KV
(q8_0/q4_0-class) + Qwen3-ASR @ Q8**. Concrete artifacts, runtime, and
config defaults are frozen in the model phase (see
`docs/v2/RESEARCH.md` §S7) — the framework ships engine-agnostic today.

## Status

- **Framework: complete.** store/service (SQLite WAL + durable queue),
  inference contracts, API/CLI/GUI, RSS builder — all merged; legacy v1
  deleted; 211 tests green (incl. end-to-end integration).
- **Model phase: pending.** ① GSQ+RCO self-quantization of Qwen3.5-9B
  (2.5 bpw primary / 2.75 bpw alt, `-mtp` build), ② KV-dtype ablation
  (bf16 vs q8_0/q4_0), ③ runtime bake-off (ExLlamaV3 vision check vs
  GGUF runtime), ④ ASR 0.6B vs 1.7B on real videos. Outcomes land as
  config defaults + one launch recipe — no code changes expected.

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

## The stack (one posture — `vra engine` prints these exact commands)

| role | model · quant | size |
|---|---|---|
| VLM (port 8000) | **Qwen3.5-9B · UD-IQ3_XXS GGUF** (`unsloth/Qwen3.5-9B-GGUF`) | 4.0 GB |
| VLM on vLLM/CUDA | Qwen3.5-9B · INT4-AWQ + `--kv-cache-dtype fp8` | ≈5.5 GB |
| ASR (port 8081) | **Qwen3-ASR-1.7B · Q8_0 GGUF** (`ggml-org`) | 1.9 GB |

`llama-server --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn` on the
VLM instance. oMLX (Apple Silicon, native MTP) serves the same 9B model
as an alternative engine. Frontier APIs ride the same wire protocol via
`VRA_ENGINE_BASE_URL` + `VRA_ENGINE_API_KEY`. Peak working set ≈ 8.4 GB
(mmap-paged); no tiers, no smaller fallback models.

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
  GPT-4o-Transcribe 15.30 on WenetSpeech); qwen3.5-9b beats GPT-5-Nano /
  Gemini-2.5-Flash-Lite on VideoMME by 9–12 pts — at 4.0 GB of weights.

## Configuration

All env vars (prefix `VRA_`, plus `BIND_ADDRESS`/`API_KEY`) are listed in
`docs/v2/SPEC.md` and rendered by the setup studio. Key ones:

| var | default | meaning |
|---|---|---|
| `VRA_ENGINE_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible server |
| `VRA_MODEL` | `qwen3.5-9b` | single served model, verified via /v1/models |
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

### Source identity compatibility

New feed and manual/API submissions share a versioned `source:v2:` SHA-256
identity of the trimmed acquisition URL or path. Query parameters, their order,
encoding, fragments and path case are preserved; feed GUID is metadata, not a
substitute for the media source. This deliberately favors a possible duplicate
for differently spelled URLs over silently collapsing different videos.

Existing legacy IDs, video UUIDs, jobs and saved artifacts are not migrated,
merged or deleted. Old query-stripped records may already contain mismatched
source metadata and summaries, so they are not automatically reused. Re-ingesting
an old source can create a new record and duplicate RSS item; historical repair
requires explicit review/reprocessing. No database schema-version bump or reset
is part of this change. Explicit processing by an existing video UUID is retained.
