"""Summarization pipeline. OWNER: agent-B. Implements contracts.Summarizer.

VlmSummarizer(config, governor, ollama):
- summarize(asset): build prompt; transcript budget = ctx - images - output;
  single pass when fits, else map-reduce (chunks ~3000 tok, map text-only think
  off temp 0.2, reduce w/ frames think=config.think).
- schema exactly: {summary: str>=40, key_points: 4..8 str, visual_highlights: 0..5 str}.
- Quality gate + ONE retry (think=low, no schema, JSON extraction) + extractive
  fallback (legacy _fallback_key_points logic ported).
- Never invent facts; transcript language; record token counts + latency + vram.
- runtime_status()/prepare_models() delegate to governor (bootstrap pulls models).
"""
