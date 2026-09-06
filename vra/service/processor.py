"""Per-video pipeline orchestrator. OWNER: agent-C. Implements contracts.SourceProcessor.

Processor(store, acquirer, transcriber, extractor, summarizer, config):
- process(video_id, force): status=acquiring; resolve meta (store.save_meta);
  subtitles() else (asr.available() and asr.transcribe(download())) — when transcript
  empty AND frames will be extracted, download happens anyway; frames via extractor;
  status=summarizing; summarizer.summarize(asset); persist transcript/frames/summary;
  status=done. Any exception -> status=failed(error=str). force skips done short-circuit.
- Overlap note: caller (queue worker) handles concurrency; GPU serialization inside
  summarizer via asyncio.Semaphore(gpu_workers).
"""
