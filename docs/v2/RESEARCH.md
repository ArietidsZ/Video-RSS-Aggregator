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

## S2.1. Iso-footprint model selection (single posture, no tiers)

Product directive: ONE low-overhead posture everywhere — no envelope
tiers. Within the ~8 GB peak envelope, prefer the largest model at the
most aggressive safe quantization over a smaller model at conservative
bits (capability-per-byte scaling; Dettmers & Zettlemoyer, k-bit
inference scaling laws, arXiv:2212.09720 — uniform-4bit optimality does
not transfer to per-layer importance-mixed quants like Unsloth Dynamic,
which shift the frontier toward 2–3 bit; AQLM/QuIP# line shows 2–3 bit
large ≥ 4-bit small at iso-memory).

Measured size ladder (unsloth GGUF):

| model · quant | size | note |
|---|---|---|
| Qwen3.5-4B Q4_K_M | 2.40 GB | previous default — CUT as capability-under-spent |
| **Qwen3.5-9B UD-IQ3_XXS** | **4.02 GB** | **single default** |
| Qwen3.5-9B UD-IQ2_M | 3.65 GB | headroom-starved alternative |
| Qwen3.5-35B-A3B UD-IQ2_M | 11.39 GB | exceeds single envelope — not offered (no tiers) |

Why 9B@UD-IQ3_XXS beats 4B@Q4_K_M at ≈ equal +1.6 GB: bf16 baseline gaps
are largest exactly on our task axis — AA-LCR 63.0 vs 57.0 (+6.0),
LongBench v2 55.2 vs 50.0 (+5.2), VideoMMMU 78.9 vs 74.1 (+4.8), MLVU
84.4 vs 82.8 — and well-tuned 3-bit mixed quants retain most of that on
doc/long-context evals. Residual quant cost is the ablation arbiter
(`vra benchmark` spot-checks vs the 4B fallback path before cutover;
HYPOTHESIS flagged, not assumed).

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

## S4.5. Quantization policy (default posture: everything that can be
# quantized IS quantized — low-compute speed is a product requirement)

Principle: quantize as hard as the accuracy-parity bar allows. WER/benchmark
retention, not tradition, sets the floor per component.

| component | single default | size | why not lower |
|---|---|---|---|
| VLM model (all engines) | **Qwen3.5-9B** | 17.9 GB bf16 | iso-footprint pick (S2.1); no smaller model anywhere in the stack |
| VLM weights (llama-server, default engine) | **UD-IQ3_XXS GGUF** (`unsloth/Qwen3.5-9B-GGUF`) | 4.02 GB | per-layer importance-mixed 3-bit; quality floor is benchmark-arbitrated |
| VLM weights (vLLM/SGLang, CUDA) | INT4-AWQ/GPTQ 9B checkpoint (community now; AngelSlim 9B recipe via configs/qwen3_5) | ≈5.5 GB | W4A16 is the lowest vLLM-native bit with solid kernels |
| VLM weights (oMLX, Apple Silicon) | MLX 3/4-bit 9B + native MTP | ≈4–5 GB | same model, same posture |
| VLM weights (frontier API) | N/A | — | provider-side |
| KV cache | llama-server `--cache-type-k q8_0 --cache-type-v q8_0` (with `--flash-attn`); vLLM `--kv-cache-dtype fp8` | ~2× KV headroom | q4 KV degrades long-context recall (AA-LCR class tasks) |
| ASR model (all engines) | **Qwen3-ASR-1.7B GGUF Q8_0** (`ggml-org/…`, official) | 1.9 GB | WER is directly sensitive to weight noise — 8-bit is the ASR floor; bf16 only if disk-free |
| frames | JPEG q=3, 768px | — | already compact |

Decode-speed rationale: LLM decoding is memory-bandwidth-bound; 4-bit
weights ≈ 2× tokens/s vs 8-bit on the same silicon, which is the single
biggest lever below GPU class. Speculative decoding: AngelSpec/Eagle3
drafts exist for Qwen3-4B/VL-4B (not yet Qwen3.5-4B) — adopt when
released; oMLX native MTP is today's Mac lever (community A/B: Qwen 27B
48–65 tok/s on M4 Max).

AngelSlim (Tencent/AngelSlim, ★1.6k, arXiv 2602.21233): ADOPTED as
checkpoint source + self-quantize recipe (FP8-Static / INT4-GPTQ / AWQ /
GPTAQ / NVFP4; zoo ships Qwen3.5-35B-Q4_K_M GGUF and Qwen3-VL FP8-Static
2B/4B/8B/32B). CUT as a runtime dependency — the app is checkpoint-
agnostic by design (`VRA_MODEL` names whatever the server serves).

## S5. Storage / queue / scheduling (engineering)

SQLite WAL + parameterized SQL + `jobs` table (retry backoff 30·2^n,
stale requeue on boot) — persistent across crashes; no redis/celery by
Occam (single-host, single-GPU). Feed polling via conditional GET
(ETag/If-Modified-Since) — RFC 7232 canonical. Idempotency: canonical
video identity (extractor:id) + content-addressed caches; reprocessing
suppressed unless forced. No paper-level controversy; ablations recorded
in ABLATED.md.

## S6. Single-stack budget (the only posture — no tiers)

| component | resident |
|---|---|
| Qwen3.5-9B UD-IQ3_XXS + q8_0 KV (32k ctx, vision) | ≈ 4.0–5.0 GB |
| Qwen3-ASR-1.7B Q8_0 GGUF (mmap-paged, sequential stage) | ≈ 1.9 GB |
| OS + Python + ffmpeg | ≈ 2.5 GB |
| **peak working set** | **≈ 8.4 GB** (mmap paging absorbs the overlap) |

One envelope, one model choice: 9B@3-bit + ASR@8-bit. Decode speed at
3-bit stays ≈ 2× the 8-bit alternative on the same silicon — the
iso-footprint directive and the low-overhead directive are the same
decision at this size ladder. Everything smaller spends capability it
could have had; everything larger breaks the single envelope (35B-A3B
UD-IQ2_M = 11.4 GB — deliberately NOT offered).

## Verdict summary

> **SUPERSEDED (2026-09-06):** envelope redefined to ~4 GB VRAM + ~4 GB RAM; quantization frontier moved to GSQ/RCO (non-uniform, same-format refinement); engine stance re-derived from live evidence. See **S7** below — S2.1's IQ3_XXS recipe, S4's engine table, and S4.5's uniform-quant policy are historical record only. The verdict table's *structural* rows (single posture, no tiers, SQLite queue) stand.

| stage | pick | runner-up kept as | evidence strength |
|---|---|---|---|
| ASR | Qwen3-ASR-1.7B @ Q8_0 | none — single posture | WER tables, multi-benchmark |
| VLM | qwen3.5-9b @ UD-IQ3_XXS (4.02 GB) | none — single posture | S2.1 iso-footprint + bf16 deltas + scaling laws |
| frames | ffmpeg scene+uniform, dHash | TransNetV2 optional extra | experimental + F1 literature |
| engine | ollama default, OpenAI-compat adapter | oMLX/vLLM/SGLang per platform | capability matrices, perf claims |
| orchestration | SQLite queue, asyncio workers | — | engineering judgment |

## S7. Model-phase dossier (researched 2026-09-06, live-verified; execution deferred)

User directive: **frameworks first, model last.** Everything below is verified via live web
inspection (HF configs/cards, GitHub repos) — zero memory-based claims. The framework
(`EngineClient` + `OpenAiAsrTranscriber` + config URLs) already covers every candidate here;
S7 outcomes require only config defaults + the `vra engine` launcher recipe.

### S7.1 Envelope
~4 GB VRAM + ~4 GB RAM total (user-stated). Supersedes S2.1/S4/S6 assumptions. "Piece of toast"
now means this envelope; 5090-class machines (user's Ninfer box) are reference, not target.

### S7.2 Architecture landscape (HF config.json, read directly)
- **Qwen3.5-9B** — `Qwen3_5ForConditionalGeneration`, dense GDN hybrid: 32 layers,
  `full_attention_interval=4` → **8 full-attn (GQA 4 KV heads × 256 head_dim) + 24 linear**;
  hidden 4096, inter 12288; native vision (image+video tokens, SigLIP-class tower ~0.45 B);
  **no audio input** (ASR stays a separate stage). BF16 = 19.31 GB (9.65 B params incl vision).
- **Qwen3.8-27B** — same family, 64 layers → 16 full-attn, hidden 5120, MTP 1-layer draft, no audio.
- Family floor is 27 B for 3.6/3.8; `Qwen3.8-Flash-Next` is a ~409 GB MoE (131 shards, measured).
  ⇒ **Qwen3.5-9B is the only GDN-hybrid + vision member that fits 4 GB.**

### S7.3 Quantization frontier: GSQ + RCO (ISTA-DASLab; arXiv 2604.18556 / 2605.00649)
- **GSQ**: Gumbel-Softmax-learned per-coordinate grid + per-group scales; layer-wise offload
  (self-quantization feasible on small VRAM); official wrapper covers `Qwen3.5/3.6 dense`.
- **RCO**: per-tensor mixed-precision allocation under exact size budget (GGUF K-Quant types).
- Deploy paths: (a) **refined GGUF K-Quant** — drop-in on GGUF runtimes; (b) **compressed-tensors
  HF → vLLM auto-detect** (verified via `Qwen3.8-27B-3Bit-GSQ`: vLLM 0.27.1 + embedding patch,
  11.83 GB w/ vision / 10.90 text-only, requires sm ≥ 80).
- Evidence (same file size/format, GSQ-refined vs unsloth UD init): 4B Q2_K_XL AIME25
  **26.67→60.00**, GPQA-D 56.06→67.17, IFEval 76.14→82.97, GSM8K 79.61→88.48, MMLU-Pro
  66.78→70.87. 8B Q2_K avg 50.03→56.28. 27B GSQ-RCO: 2.50 bpw/8.4 GB recovery **100.3%**,
  2.75 bpw/9.3 GB **101.8%**; matched-size IQ2_XS beats UD-IQ2_S by +10.00 AIME25 / +8.59 GPQA.
- **1.25-bit verdict:** no public 27B artifact (GSQ-RCO floor 2.50 bpw). 1.25-bit releases exist
  only for 1.8–3 B translation models (AngelSlim Hy-MT). 27B@1.25 ≈ 4.2 GB is not purchasable —
  do not plan around it.

### S7.4 KV policy (GDN hybrid)
Only ¼ of layers carry KV. Qwen3.5-9B bytes/token: bf16 32 KiB, int8/fp8 16 KiB,
**K8V4-type (q8_0 K + q4_0 V) 12 KiB** → @24k ctx: 0.75 / 0.38 / 0.28 GB.
- K8V4 is a first-class KV representation in Ninfer (`bf16/int8/fp8/nvfp4/k8v4`; perf baseline
  INT8 group-64). User reports K8V4 > INT8/FP8 quality on their 5090 deployment — **to be
  confirmed by our own PPL + task ablation before freeze**, not accepted on authority.

### S7.5 Engine matrix vs 4 GB envelope (all live-verified)
| Engine | Verdict | Evidence |
|---|---|---|
| Ninfer | dead @4GB / reference for 5090 | hard `sm_120a`, Linux-only, registered artifacts (3.6/3.8-27B gwi+nvfp4, 35B-A3B gwi); 27B gwi 16.96 GiB; AIME25 96.67 / GPQA-D 87.37 / ERQA 66.25 (vision) / IFBench 77.67; NVFP4 ≥ gwi on 4/5 benchmarks at 2.4× decode |
| vLLM (compressed-tensors) | ≥12 GB machines only | GSQ path verified; runtime footprint exceeds 4 GB VRAM/RAM for 9B-class VLM |
| ExLlamaV3 / EXL3 | alternative, pending vision check | official `turboderp/Qwen3.5-9B-exl3` + 27B full ladder; community 2.0/2.5 bpw; **vision/mmproj support unverified** |
| GGUF runtime | **arithmetic-forced executor @4GB** | native GSQ K-Quant target; mtmd vision verified for family (27B card); MTP draft builds (`-mtp`) verified; smallest overhead. User's kernel-efficiency objection noted — irrelevant at C=1-2 batch summarization |

### S7.6 ASR (WER table read from Qwen3-ASR-0.6B card)
- **0.6B**: zh WenetSpeech 5.97\|6.88, en Librispeech 2.11\|4.55, GigaSpeech 8.88 (beats
  Whisper-large-v3 9.76 and Doubao 9.55).
- **1.7B**: zh 4.97\|5.88, en 1.63\|3.38, GigaSpeech 8.45. GPT-4o-Transcribe zh 15.30\|32.27.
- Both crush frontier APIs on zh; 0.6B trails 1.7B by ~1 WER pt. 0.6B@Q8 ≈ 0.7 GB RAM vs
  1.7B@Q8 1.9 GB → **0.6B default** under 4 GB RAM, pending a 20-video spot check at model phase.

### S7.7 Model-phase execution plan (deferred; nothing here blocks the framework)
1. **Artifacts**: GSQ(+RCO) on Qwen3.5-9B → 2.5 bpw primary / 2.75 bpw alt, init from UD GGUF,
   `-mtp` build included, allocation dumps archived. One-time (local offloaded run or rented GPU).
2. **KV ablation**: bf16 vs q8_0/q4_0 vs q8_0/q8_0 — fixed PPL corpus + our summarization task
   set; freeze lowest-bit within noise.
3. **Runtime bake-off**: ExLlamaV3 vision-support check first; else GGUF runtime. Single posture.
4. **ASR pick**: 0.6B vs 1.7B on real videos (WER + downstream summary quality), quantize Q8.
5. Set config defaults; `vra engine` materializes the chosen launch recipe.

VRAM ledger @2.5 bpw, 24k ctx: weights 3.02 + KV(q8_0/q4_0) 0.28 + mmproj Q8 0.45 + CUDA ctx
0.2-0.4 ≈ **3.95-4.15 GB** (knobs: RCO→2.25-2.4 bpw, mmproj→CPU, ctx→16k). RAM ledger:
app 0.3-0.5 + ffmpeg/yt-dlp peak 0.8 + ASR-0.6B 0.9 + runtime 0.3 ≈ **2.3-2.5 GB**.
