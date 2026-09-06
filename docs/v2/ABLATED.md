# Occam's razor ablations — v2

Every abstraction considered, and the verdict, with the evidence that
decided it. Revisit only when evidence changes.

| Candidate | Verdict | Evidence / reasoning |
|---|---|---|
| APScheduler / celery / redis for jobs | **CUT** | Single process, SQLite already durable. An asyncio worker loop on a `jobs` table gives persistence + crash recovery (requeue_stale) with zero new deps. |
| PySceneDetect / opencv for shot detection | **CUT** | ffmpeg `select=gt(scene,T)` + `metadata=print` gives per-frame scores in the same decode pass; opencv would add a ~90MB dep and a second decode. |
| imagehash / Pillow for perceptual dedup | **CUT** | dHash (9x8 grayscale) is 20 lines and computed by the same ffmpeg binary via rawvideo pipe. Identical hamming semantics. |
| numpy for frame math | **CUT** | Nothing needs vectorization at ≤16 frames/video. |
| Multi-model ensemble / LLM-as-judge | **CUT** | One GPU; ensemble multiplies VRAM+latency by N for marginal gain at 4B scale. Replaced by: schema-constrained output + quality gate + think-escalation retry + extractive fallback. |
| Fixed model size hints table | **REPLACED** | v1 hardcoded `MODEL_SIZE_HINT_MB` per tag; v2 queries `/api/tags` for real sizes, name-parse only as fallback. |
| `keep_alive: 0` per request (v1 behavior) | **BUG, REPLACED** | Reloaded weights every request. v2 pins with `VRA_KEEP_ALIVE` (default 15m) during bursts; governor unloads on idle. |
| Fixed `VRA_CONTEXT_TOKENS` (v1: 3072) | **BUG, REPLACED** | 16k-char transcripts + 5 images silently exceeded 3072 ctx. v2 computes ctx from payload (chars→tokens incl. CJK, image tokens from JPEG SOF dims) and map-reduces long transcripts. |
| Separate CPU and GPU job queues | **CUT (unified)** | `cpu_workers` process workers + GPU semaphore (`gpu_workers`) inside the summarizer. Downloads overlap inference; VRAM contention impossible by construction. Two queues would add scheduling complexity for zero throughput gain. |
| `thumbnail` filter in uniform branch | **CUT** | Uniform grid already guarantees temporal coverage; thumbnail adds a buffer stage without measurable quality gain at 8 frames. Re-evaluate with bench data. |
| Transcription (ASR) always on | **OPT-IN dep** | faster-whisper is an optional extra (`pip install .[asr]`); runtime autodetects. Subtitles remain the zero-cost path. |
| ORM / migrations framework | **CUT** | One frozen DDL script + `schema_version` row; aiosqlite parameterized SQL. |
| `-frame_pts true` filename timestamps | **DEAD** | Verified broken on ffmpeg 8.1.2 (filenames stay sequential). Replaced by `-fps_mode passthrough` + `metadata=print` scene log (verified: file k ↔ log line k) and analytic uniform ts. |
| Multi-engine ladder (Ollama native + governor + faster-whisper + mlx-audio + extractive fallback) | **CUT — no-fallback mandate** | One OpenAI-compatible engine path + one ASR wire protocol; quantized defaults keep the same hardware faster. Everything else was a second way to do the same thing. |
| bf16/fp16 default checkpoints | **CUT — quantize by default** | W4 VLM (AWQ/Q4_K_M/MLX-4bit) + q8/fp8 KV + ASR Q8_0: ≈2× decode speed on bandwidth-bound silicon, ≈2× KV headroom; accuracy retention is benchmark-bounded (RESEARCH S4.5). |
| AngelSlim as runtime dependency | **CUT (code) / ADOPTED (checkpoint source)** | App is checkpoint-agnostic; AngelSlim's INT4/FP8 recipes + zoo are the recommended quantization provenance for CUDA deployments. |
| ASR WER vs speed trade at Q4 | **REJECTED for ASR** | WER is directly noise-sensitive; ASR Q8_0 stays (1.9 GB). VLM takes the 3-bit hit instead — summarization tolerates it, transcription does not. |
| Tiered envelopes (4GB/8GB/16GB model ladders) | **CUT — single posture** | One stack everywhere: qwen3.5-9b @ UD-IQ3_XXS + ASR @ Q8_0. Tier ladders were configuration surface without evidence of use; iso-footprint scaling (S2.1) says spend the same bytes on a bigger model. |
| Qwen3.5-4B as default | **CUT** | bf16 task-axis deltas are +5–6 pts for 9B exactly where this product lives (AA-LCR/LongBench/VideoMMMU); 3-bit mixed quants retain most at +1.6 GB. Benchmark-arbitrated before cutover. |
| TransNetV2 shot detector | **CUT from core, optional extra** | F1 0.868 (BBC) leads classical ~0.1-0.2, but torch wheel breaks toast tier; uniform+scene blend + dHash bounds the damage. Re-evaluate via benchmark. |
| Single Ollama-only engine | **REPLACED by engine matrix** | oMLX (Apple Silicon continuous batching + KV tiering + MTP), vLLM v0.28 (Qwen3.5 + Qwen3-ASR arch support, guided_json), SGLang(-Omni) — one OpenAI-compat adapter reaches them all. |
| Legacy root shims (`adapter_*.py`, `service_*.py`, `core_config.py`, `video_rss_aggregator/`) | **DELETE at cutover** | v2 `vra/` package supersedes; shims existed only for compatibility. |
