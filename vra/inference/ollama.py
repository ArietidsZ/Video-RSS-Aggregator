"""Ollama HTTP client. OWNER: agent-B. Port + extend legacy service_ollama.

- OllamaClient: version(), tags() (name->size_bytes map, replaces hardcoded hints),
  ps() (running models + vram), chat() with model, messages, images(list[base64]),
  format schema|None, think bool|None or level str, options(num_ctx, num_predict,
  temperature), keep_alive str. Retries: 3x exp backoff on connect/timeout/5xx.
- is_thinking_compatible: think + format(schema) combined is unsupported upstream
  (ollama#10538): chat() must raise ValueError if think not in (None, False) and
  format is a dict. Callers pass think=False with schema, or think=level without.
"""
