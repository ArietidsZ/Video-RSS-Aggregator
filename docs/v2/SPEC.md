# VRA v2 — Production Spec (frozen contract)

Single source of truth for the v2 rebuild. Agents must not deviate from
signatures/schemas here; propose changes to the lead instead.

## Goals

Local-first video RSS aggregator: poll feeds → acquire video (metadata,
subtitles, ASR fallback, frames) → summarize with local VLM (Ollama) →
publish RSS. Production-grade: durable job queue, scheduler, idempotency,
security hardening, observability, near-optimal GPU/CPU utilization.

## Non-goals (ablated by Occam)

- No APScheduler / celery / redis. asyncio-native queue + loop only.
- No opencv / PySceneDetect / imagehash / numpy. ffmpeg filters + Pillow-free
  pure-python dHash (16-hex via 9x8 grayscale diff) only.
- No multi-model ensemble/judge. Single routed model + think escalation.
- No ORM. aiosqlite + fixed schema below.

## Layout (strict file ownership)

```
vra/
  __init__.py          # __version__ = "2.0.0"
  types.py             # domain dataclasses/enums (FROZEN by lead)
  config.py            # Config dataclass + from_env (FROZEN by lead)
  contracts.py         # Protocols wiring services together (FROZEN by lead)
  runtime.py           # composition root (LEAD only)
  platform/            # OWNER: agent-A
    binaries.py  ytdlp.py  media.py  subtitles.py  asr.py
  inference/           # OWNER: agent-B
    ollama.py  governor.py  summarize.py
  store/               # OWNER: agent-C
    schema.py (FROZEN: SQL DDL string)  db.py  repos.py  queue.py
  service/             # OWNER: agent-C
    ingester.py  processor.py  scheduler.py
  api/                 # OWNER: agent-D
    app.py  rss.py  gui.py  setup_assets.py
  cli.py               # OWNER: agent-D
  templates/, static/  # OWNER: agent-D (port from video_rss_aggregator/)
tests/v2/              # each agent adds tests/v2/<area>_test.py
```

Root legacy modules (`core_config.py`, `service_*.py`, `adapter_*.py`,
`cli.py`, `video_rss_aggregator/`) stay untouched during build; deleted at
cutover by lead. Nobody imports from them except agent-D porting GUI assets.

## Domain types (vra/types.py — see file for exact fields)

VideoRef, VideoMeta, TranscriptSeg, Transcript, FrameSet/Frame, Summary,
JobKind/JobState/Job, FeedRecord, VideoRecord, IngestResult.

## Config keys (vra/config.py — full table)

All env vars prefixed `VRA_` except `BIND_ADDRESS`, `API_KEY`.

| key | default | meaning |
|---|---|---|
| BIND_ADDRESS | 127.0.0.1:8080 | uvicorn bind |
| API_KEY | none | bearer/x-api-key auth for mutating endpoints |
| VRA_DATA_DIR | .data | storage root |
| VRA_DATABASE_PATH | $DATA_DIR/vra.db | sqlite path |
| VRA_OLLAMA_BASE_URL | http://127.0.0.1:11434 | |
| VRA_MODELS | qwen3.5:4b,qwen3.5:2b,qwen3.5:0.8b | priority list, first fit wins |
| VRA_PREFER_MLX | auto | auto=on darwin, appends `-mlx` variant if present in /api/tags |
| VRA_THINK | off | off\|low\|medium\|high\|max — base think level |
| VRA_KEEP_ALIVE | 15m | model pin duration during bursts |
| VRA_VRAM_BUDGET_MB | 8192 | total model+KV budget |
| VRA_RESERVE_MB | 768 | safety reserve |
| VRA_MAX_FRAMES | 8 | frames sent to VLM |
| VRA_FRAME_MODE | auto | auto\|scene\|uniform |
| VRA_SCENE_THRESHOLD | 0.3 | select=gt(scene,T) 0..1 |
| VRA_FRAME_MAX_DIM | 768 | max(w,h) of extracted JPEG |
| VRA_MAX_TRANSCRIPT_CHARS | 24000 | hard cap |
| VRA_CTX_TOKENS | auto | auto = computed from payload, else fixed |
| VRA_MAX_OUTPUT_TOKENS | 1024 | num_predict |
| VRA_ASR | auto | auto\|off — transcribe when no subs |
| VRA_ASR_BACKEND | auto | auto\|mlx\|vllm\|llama_cpp\|faster_whisper (chain, RESEARCH.md S1) |
| VRA_ASR_MODEL | Qwen/Qwen3-ASR-1.7B | Qwen3-ASR model (0.6B = toast floor) |
| VRA_ASR_WHISPER_MODEL | small | faster-whisper fallback model |
| VRA_ENGINE | auto | auto\|ollama\|openai — auto=ollama unless VRA_ENGINE_BASE_URL set |
| VRA_ENGINE_BASE_URL | none | OpenAI-compatible base (vLLM/SGLang/oMLX/llama-server/frontier API) |
| VRA_ENGINE_API_KEY | none | bearer for VRA_ENGINE=openai |
| VRA_ASR_DEVICE | auto | auto\|cpu\|cuda |
| VRA_POLL_MINUTES | 15 | default feed poll interval |
| VRA_CPU_WORKERS | 2 | download/frames queue workers |
| VRA_GPU_WORKERS | 1 | summarize queue workers |
| VRA_TIMEOUT_DOWNLOAD | 1800 | s |
| VRA_TIMEOUT_INFER | 900 | s |
| VRA_ALLOW_LOCAL_PATH | false | accept local file paths via API (CLI always true) |
| VRA_ALLOW_PRIVATE_NET | false | allow private/loopback feed URLs |
| VRA_RSS_TITLE/LINK/DESCRIPTION | as v1 | feed metadata |

## DB schema (vra/store/schema.py — FROZEN, see file)

feeds, videos, transcripts, frames, summaries, jobs, schema_version.
Canonical identity: `videos.canonical_id` = yt-dlp extractor:key:id or
sha1(source_url) when unknown. All content-addressed work keys off it.

## Pipeline

1. **ingest** (service/ingester.py): fetch feed (ETag/If-Modified-Since via
   httpx headers on GET), map entries → upsert videos(status=pending) →
   enqueue Job(kind=process) per new/failed video. Feed upsert with
   poll_interval_s. Dedup by canonical_id — never reprocess done videos
   unless job forced=True.
2. **process** (service/processor.py): given video →
   acquire (platform.ytdlp: meta+subs, download only if frames needed) →
   transcript (subtitles.py; else ASR chain per RESEARCH.md S1:
   Qwen3-ASR via mlx-audio / vLLM toolkit / llama-server GGUF, then
   faster-whisper int8 fallback) →
   frames (media.py single decode pass: scene-select + uniform-grid blend,
   dHash dedup, quality rank, cap VRA_MAX_FRAMES) →
   summarize (inference.summarize) → persist rows, mark done. CPU stage and
   GPU stage overlap: processor awaits gpu semaphore only around inference.
3. **queue** (store/queue.py): jobs table; asyncio worker pool
   (CPU_WORKERS consume kind=process acquire stage via two-phase states
   queued→running→done/failed; attempts backoff 30s*2^n, max 3;
   visibility: crashed running jobs older than 30min are requeued at boot).
4. **scheduler** (service/scheduler.py): per-feed asyncio task, sleep
   poll_interval + jitter(≤10%), enqueue ingest job. Single global loop.

## Summarization contract (inference/summarize.py)

- Token estimate: latin ≈ chars/3.6, CJK ≈ chars/1.4 (per-char check).
- num_ctx = ceil((prompt_tokens + image_tokens + output + 256)/512)*512,
  clamped [1024, 32768]. image_tokens ≈ Σ ceil(w/28)*ceil(h/28)/4 per frame.
- Transcript ≤ budget → single pass with frames. Else map-reduce: chunk
  ~3000 tokens, map (text-only, think off, cheap), reduce with frames.
- format=schema JSON; think=VRA_THINK for the final pass only.
- Quality gate: summary ≥ 40 chars or ≥2 key points; else ONE retry with
  error feedback appended, think=low, NO format schema (manual JSON
  extraction). Then fallback: extractive key points from transcript.
- Every attempt records model, tokens, latency into summaries row.

### Engine matrix (RESEARCH.md S4)

- `VRA_ENGINE=ollama` (default): OllamaClient + VRAM governor as specified;
  `think`+dict-format ValueError guard stands (ollama#10538).
- `VRA_ENGINE=openai`: OpenAI-compatible `/v1/chat/completions` with
  `response_format: {type: "json_schema", json_schema: ...}` and base64
  image parts — covers vLLM / SGLang / oMLX / llama-server / frontier APIs
  (base_url + api_key). VRAM governance is server-side; model routing
  still uses VRA_MODELS against `/v1/models`. Implemented by the lead
  post-merge (`vra/inference/openai_engine.py`); agent-B does NOT
  implement it.

Evidence for every stage decision: docs/v2/RESEARCH.md.

## API (api/app.py) — all JSON errors {"detail": str}

- GET /health → {"status":"ok","version":...}
- GET /setup/config | /setup/diagnostics ; POST /setup/bootstrap (port v1)
- POST /feeds {url, poll_minutes?} → 201 FeedRecord ; GET /feeds ;
  DELETE /feeds/{id} (and its pending jobs)
- POST /ingest {url} → enqueues Job(kind=ingest) → {job_id}
- POST /process {source_url | video_id, force?} → {job_id}
- GET /jobs?state=&kind=&limit= ; GET /jobs/{id}
- GET /videos?limit=&feed_id= ; GET /videos/{id} (incl. latest summary,
  transcript, frames)
- GET /rss?limit=20 (media:thumbnail of first frame, enclosure meta)
- GET /metrics {jobs_by_state, videos_by_status, vram_mb, model}
- Mutating endpoints require auth when API_KEY set. SSRF guard on feed/url
  fetch: scheme http/https, deny private/loopback IPs unless
  VRA_ALLOW_PRIVATE_NET. Local paths only when VRA_ALLOW_LOCAL_PATH
  (API) or CLI (always).

## CLI (cli.py, entry `vra`)

serve | ingest URL | process URL [--force] | status | bootstrap |
benchmark URL (frames uniqueness + latency scene vs uniform) | cleanup

## Acceptance (per agent)

- `python -m pytest tests/v2 -q` green with your tests; ruff clean.
- No file outside your ownership list modified.
- Public functions fully typed, docstring on every module.
- Async everywhere; no blocking IO on event loop (subprocess + to_thread).
- Windows (nt) and macOS both supported; no POSIX-only calls.
