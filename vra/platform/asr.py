"""faster-whisper ASR fallback. OWNER: agent-A. Implements contracts.Transcriber.

- available(): importlib.util.find_spec("faster_whisper") is not None (add dep only
  if present; never import at module top).
- transcribe(): WhisperModel(asr_model, device=auto(cpu/cuda), compute_type=int8 on
  cpu, float16 on cuda) in asyncio.to_thread; vad_filter=True; map segments ->
  Transcript(source="asr", lang=info.language). Cache transcript text per
  canonical_id in DB is caller's job.
"""
