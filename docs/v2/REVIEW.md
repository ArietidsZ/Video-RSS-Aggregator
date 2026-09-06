# Merge-day review checklist (lead)

Per-slice review pass on each branch before `git merge --no-ff`, then
integration. Any finding → fix on the agent's branch by the agent (prompt
follow-up) or by lead if trivial.

## A — platform
- [ ] Every subprocess call: absolute binary path, timeout, stderr captured, exec (no shell)
- [ ] canonical_id stable across runs; filesystem-sanitized variant capped at 120 chars
- [ ] download idempotent (existing file short-circuit), merge output path from --print
- [ ] single decode pass verified in code (one -i, filter_complex, two -map)
- [ ] -fps_mode passthrough on both image2 outputs; scene.log parse ↔ file count asserted
- [ ] dHash: 72-byte rawvideo read, hamming ≤ 6, tie-break by scene score then ts
- [ ] subtitle rolling-dedup + SRT/VTT both handled; max_chars clamp at boundary
- [ ] ASR lazy import; no module-level faster_whisper import
- [ ] Tests: skipif no ffmpeg; no network anywhere; ytdlp parse tested via monkeypatch

## B — inference
- [ ] think truthy + dict format → ValueError (ollama#10538 guard)
- [ ] retries only on connect/timeout/5xx; 4xx immediate; backoff 1/2/4s
- [ ] governor: tags() real sizes; name-parse fallback; MLX swap rules; kv heuristic documented
- [ ] ctx_for rounding to 512, clamp [1024, 32768]; image tokens from JPEG SOF (no Pillow)
- [ ] map-reduce trigger math; chunking at sentence boundaries ~3000 tokens
- [ ] quality gate + single think-escalation retry + extractive fallback; error surfaced not raised
- [ ] gpu_slot() semaphore around EVERY chat call; close() cascades to httpx client

## C — store + service
- [ ] parameterized SQL everywhere; JSON round-trips; ISO-8601 UTC timestamps
- [ ] claim() atomic under lock; RETURNING row; attempts increment
- [ ] backoff run_at = now + 30*2^attempts; terminal failed state
- [ ] upsert_video_ref (record, created) semantics; done-short-circuit + force
- [ ] ingest: guid-vs-url canonical rule identical to app.py /process rule
- [ ] HttpFeedFetcher: conditional GET, 304 path, SSRF private-net guard, datetime fallbacks ported
- [ ] scheduler jitter in [1.0, 1.1); start/stop idempotent
- [ ] v1 db detection + .v1.bak rename once

## D — api + cli + gui
- [ ] auth dependency on all POST/DELETE; GETs open; 401 shape {"detail":"unauthorized"}
- [ ] SSRF guard on body URLs; /media path traversal blocked (resolve + is_relative_to)
- [ ] RSS: atom self link, media ns, guid isPermaLink=false, feedparser validates in test
- [ ] create_app(runtime) duck-typed facade; no build_runtime import at module level
- [ ] CLI serve builds runtime via vra.runtime.build_runtime; other commands hermetic
- [ ] GUI port: v2 config field names in env table; setup endpoints adapted
- [ ] templates/static copied into vra/; packaged via package-data

## Integration (lead)
- [ ] merge order: A, B, C, D (disjoint files; expect zero conflicts)
- [ ] runtime.py compiles against actual class signatures; close() cascade correct
- [ ] tests/v2/test_integration.py: fake-fetcher ingest → queue workers → done video → summary row
- [ ] full suite green (legacy suite may fail pre-cutover — ignored in CI, removed after)
- [ ] ruff clean on vra + tests
- [ ] README.v2.md → README.md; delete legacy shims; pyproject py-modules cleanup
- [ ] E2E sanity on real feed + local media file with live Ollama (manual, documented)
