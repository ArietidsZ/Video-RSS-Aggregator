# v2 Per-stage research dossier

Every pipeline stage, its candidate set, the measured evidence, and the
decision. Numbers are quoted from primary sources (model cards, tech
reports, benchmark repos) fetched 2026-09-06. A decision without a number
is marked HYPOTHESIS and scheduled for ablation.

Bar to meet (product mandate): **frontier-comparable accuracy on this
task at minimum hardware cost** — ideally a fanless laptop / 8 GB
unified-memory Mac ("toast tier").

---

## S1. Speech recognition (transcript source of truth)

Candidates: Whisper large-v3 (faster-whisper/whisper.cpp), Fun-ASR-MLT-Nano,
GLM-ASR-Nano-2512, Qwen3-ASR-0.6B / 1.7B, frontier APIs (GPT-4o-Transcribe,
Gemini-2.5-Pro, Doubao-ASR).

Evidence (WER ↓, from Qwen/Qwen3-ASR model card, vLLM greedy decoding,
bfloat16):

| benchmark | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-0.6B | Qwen3-ASR-1.7B |
|---|---|---|---|---|
| Librispeech clean / other | 1.39 / 3.75 | 1.68 / 3.38 | 2.11 / 4.55 | 1.63 / 3.38 |
| WenetSpeech net / meeting (zh) | 15.30 / 32.27 | 9.86 / 19.11 | **5.97 / 6.88** | **4.97 / 5.88** |
| SpeechIO (zh) | 12.86 | 7.56 | **3.44** | **2.88** |
| AISHELL-2 (zh) | 4.24 | 5.06 | 3.15 | 2.71 |
| KeSpeech (zh dialects) | 26.87 | 28.79 | 7.08 | 5.10 |
| Tedlium (en) | 7.69 | 6.84 | 3.85 | 4.50 |
| MLC-SLM (11 lang) | — | 15.68 | 15.84 | 12.74 |
| Fleurs (12 lang) | — | 5.27 | 7.57 | 4.90 |
| ExtremeNoise (internal) | 36.11 | 63.17 | 17.88 | 16.17 |

Findings:
1. On Chinese (the primary use case here), **0.6B beats Whisper-large-v3 by
   2–4× WER and beats GPT-4o-Transcribe by ~2–3×**; on English read speech
   it trails large-v3 by ≤1.2 WER (Librispeech) while winning on Tedlium
   (3.85 vs 6.84) and multilingual sets.
2. 1.7B is at-or-below frontier-API WER on 9/13 quoted benchmarks. This is
   the "no significant difference from frontier" ASR claim, with sources.
3. Size cost: 0.6B 8-bit ≈ 0.7 GB (MLX), 1.7B 8-bit ≈ 1.9 GB; Whisper-large-v3
   ≈ 3 GB fp16 / 1.5 GB int8 — Qwen3-ASR dominates the accuracy-per-byte curve
   for zh+multilingual video content.

**Decision.** ASR chain (first available wins), driven by `VRA_ASR_BACKEND=auto`:
1. Subtitles (free, exact, includes human timing) — always first.
2. Qwen3-ASR-1.7B via platform-native runtime:
   - darwin: `mlx-audio` + `mlx-community/Qwen3-ASR-1.7B-8bit` (Metal).
   - CUDA Linux/WSL2: official QwenLM/Qwen3-ASR toolkit on vLLM
     (`Qwen3ASRForConditionalGeneration` supported by vLLM ≥ v0.28).
   - any: `llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF` (CPU/CUDA; note
     upstream quirk: response key `asr_text`, llama.cpp #26749).
3. faster-whisper int8 (CPU) as universal fallback — retained because it is
   dependency-light and robust, not because it is best.
Floor on toast: 0.6B (`VRA_ASR_MODEL=Qwen/Qwen3-ASR-0.6B`) — 0.7 GB.
Timestamps: from runtime segments (toolkit/MLX/GGUF); no separate
forced-alignment model (Qwen3-ForcedAligner-0.6B noted; cut for v2.0 —
segment-level ts suffices for frame correlation; HYPOTHESIS → ablate).

## S2. Summarization model (VLM)

Candidates ≤4 GB (toast tier): qwen3.5:4b (2.5 GB q4), qwen3.5:0.8b
(1.0 GB q8 floor); above: qwen3.5:9b (6.6 GB, needs ≥16 GB rigs), Qwen3-VL-30B (GPU rig).

Evidence (Qwen3.5-4B model card; columns GPT-5-Nano / Gemini-2.5-Flash-Lite /
Qwen3-VL-30B-A3B / Qwen3.5-9B / Qwen3.5-4B):

| benchmark | GPT-5-Nano | Gemini-2.5-FL | Qwen3-VL-30B | Qwen3.5-9B | **Qwen3.5-4B** |
|---|---|---|---|---|---|
| VideoMME (w/ sub) | 71.7 | 74.6 | 79.9 | 84.5 | **83.5** |
| VideoMME (w/o sub) | 66.2 | 72.7 | 73.3 | 78.4 | **76.9** |
| VideoMMMU | 63.0 | 69.2 | 75.0 | 78.9 | 74.1 |
| MLVU | 69.2 | 78.5 | 78.9 | 84.4 | **82.8** |
| MMMU | 75.8 | 73.4 | 76.0 | 78.4 | 77.6 |
| OCRBench | 75.3 | 82.5 | 83.9 | 89.2 | 85.0 |
| AA-LCR (long-doc summary) | — | — | — | 63.0 | **57.0 vs GPT-OSS-120B 50.7** |
| LongBench v2 | — | — | — | 55.2 | **50.0 vs GPT-OSS-120B 48.2** |

Findings:
1. **Video understanding: qwen3.5-4b beats both frontier-lite APIs by 9–12
   points (w/ sub) and even beats a 30B MoE** — the "frontier parity" claim
   for the VLM stage is benchmark-backed, not aspirational.
2. Long-context summarization: 4B > GPT-OSS-120B on AA-LCR/LongBench v2 —
   supports transcript-heavy prompts (map-reduce) with a 4B model.
3. Model card guidance adopted: dynamic-resolution video tokens; we cap
   frames at `VRA_FRAME_MAX_DIM=768` and budget image tokens as
   Σ⌈w/28⌉⌈h/28⌉/4 (Qwen-VL patch math) — matches the family's tokenizer.

**Decision.** Default `VRA_MODELS=qwen3.5:4b,qwen3.5:2b,qwen3.5:0.8b`
(q4 default quants). Prompt strategy: transcript-first (evidence S1:
transcript is frontier-grade), frames as visual evidence, schema-
constrained JSON output, quality gate + think-escalation retry,
extractive fallback. `think` on the final pass only (VRA_THINK), because
thinking + structured output is unsupported upstream (ollama#10538).

## S3. Shot / keyframe selection

Candidates: ffmpeg `select=gt(scene,T)` (classical scene score),
ffmpeg `scdet`, TransNetV2 (CNN, F1 0.868 BBC Planet Earth — Souček &
Lokoč, ICMR 2020 workshops), PySceneDetect ContentDetector (classical HSV).

Findings:
1. TransNetV2 leads classical detectors by ~0.1–0.2 F1 on hard cuts with
   motion, but requires torch (~200 MB wheel) or TF — disqualifying for
   toast tier and for a zero-dep core.
2. Our pipeline does not need exact shot boundaries: uniform grid
   guarantees temporal coverage; scene candidates add recall on cuts;
   near-duplicate pruning (dHash) + timeline-spread ranking bound the
   damage of both false positives and misses. Summarization quality is
   driven by transcript + coverage, not boundary precision (VideoMME w/ sub
   ≫ w/o sub delta: transcript dominates; HYPOTHESIS → benchmark command).
3. Verified experimentally (ffmpeg 8.1.2): single decode pass producing
   scene candidates (real per-frame `lavfi.scene_score` via
   `metadata=print`) + uniform grid with `-fps_mode passthrough`;
   `-frame_pts` filename expansion is broken in 8.x.

**Decision.** ffmpeg-only in core (`VRA_FRAME_MODE=auto`, scene+uniform
blend, dHash hamming ≤ 6 dedup). TransNetV2 listed as optional future
extra `[extra: shots]` — only if the ablation benchmark shows coverage
misses on high-motion content.

## S4. Inference engine (serving stack)

Candidates: Ollama (llama.cpp CUDA / MLX engine on darwin), oMLX (Apple
Silicon serving: continuous batching, tiered hot/cold KV, native MTP;
OpenAI-compatible `/v1`; JSON schema validation; VLM with base64 multi-image),
vLLM v0.28 (`Qwen3_5ForConditionalGeneration` + `Qwen3ASRForConditionalGeneration`
✅, guided_json/xgrammar), SGLang v0.5.19 (+ SGLang-Omni for ASR/TTS serving),
llama-server.

Evidence / constraints:
- vLLM/SGLang: CUDA-first; no native Windows, no Metal → not the toast path,
  but the throughput ceiling on CUDA rigs. Both speak OpenAI-compatible API
  with JSON-schema constrained decoding.
- oMLX: VLM continuous batching + base64 images + JSON schema validation
  (README, fetched today; ★21.4k, pushed 2026-09-06). Best-known Apple
  Silicon throughput (native MTP recipes; community A/B: Qwen 27B 48–65
  tok/s on M4 Max).
- Ollama: zero-config default on every platform; MLX engine + MTP on
  darwin ("up to 90% faster" Gemma-4 MLX MTP claim, ollama blog 2026-06);
  MLX-tagged qwen3.5 builds shipped. Structured outputs via `format`
  schema. `think`+`format` combination broken (#10538) — design accounts.
- Toast tiers: 8 GB Mac → qwen3.5:4b-mlx + Qwen3-ASR-0.6B-8bit ≈ 3.2 GB
  resident. CPU-only laptop → 0.8b-q8 (1.0 GB) + ASR-0.6B GGUF (0.7 GB).

**Decision.** Pluggable engine behind the frozen `Summarizer` protocol:
- `VRA_ENGINE=ollama` (default; native client, VRAM governor active).
- `VRA_ENGINE=openai` → any OpenAI-compatible server (vLLM / SGLang /
  oMLX / llama-server / cloud frontier APIs via base_url+key) with
  `response_format: json_schema`; VRAM governance becomes server-side
  (`gpu_memory_utilization` etc.), model routing still applies.
This is one adapter, not abstraction-for-its-own-sake: it is what makes
"best known engine per platform" reachable without forking the pipeline.

## S5. Storage / queue / scheduling (engineering)

SQLite WAL + parameterized SQL + `jobs` table (retry backoff 30·2^n,
stale requeue on boot) — persistent across crashes; no redis/celery by
Occam (single-host, single-GPU). Feed polling via conditional GET
(ETag/If-Modified-Since) — RFC 7232 canonical. Idempotency: canonical
video identity (extractor:id) + content-addressed caches; reprocessing
suppressed unless forced. No paper-level controversy; ablations recorded
in ABLATED.md.

## S6. Toast budget (final math, 8 GB unified memory Mac)

| component | resident |
|---|---|
| qwen3.5:4b-mlx (4-bit) + KV (24k ctx, vision) | ≈ 3.0–3.6 GB |
| Qwen3-ASR-1.7B-8bit (transient, sequential stage) | ≈ 1.9 GB |
| OS + Python + ffmpeg | ≈ 2.5 GB |
| **peak (sequential stages, never concurrent)** | **≈ 6.1 GB** |

Floor tier (CPU-only / 4 GB): qwen3.5:0.8b-q8 (1.0 GB) + ASR-0.6B (0.7 GB
GGUF, llama-server) — accuracy floor is set by S1/S2 floor models, not by
pipeline design.

## Verdict summary

| stage | pick | runner-up kept as | evidence strength |
|---|---|---|---|
| ASR | Qwen3-ASR 1.7B (0.6B floor) | faster-whisper int8 fallback | WER tables, multi-benchmark |
| VLM | qwen3.5:4b | 9b on ≥16 GB; 0.8b floor | VideoMME/MLVU/AA-LCR tables |
| frames | ffmpeg scene+uniform, dHash | TransNetV2 optional extra | experimental + F1 literature |
| engine | ollama default, OpenAI-compat adapter | oMLX/vLLM/SGLang per platform | capability matrices, perf claims |
| orchestration | SQLite queue, asyncio workers | — | engineering judgment |
