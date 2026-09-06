"""VRAM-aware model routing + pinning. OWNER: agent-B.

- Governor(config, ollama): refresh() pulls tags()+ps(); effective_model_names()
  applies VRA_PREFER_MLX (darwin + '-mlx' present in tags -> prefer).
- pick(need: {prompt_tokens, image_tokens, output_tokens}) -> ModelPlan(model,
  num_ctx, loaded, reason): budget = vram_budget - reserve - current_vram;
  per candidate: file_size from tags (fallback parse '4b'->4e9 etc), kv_bytes ~
  2 * num_ctx * layers_estimate... implement kv estimate as
  kv_gb = 0.12 * num_ctx/1024 * (params_b/4) heuristic, documented.
- pin(model)/unpin(): chat with keep_alive=config.keep_alive / "0" empty messages.
- ctx_for(payload) helper: ceil((prompt+images+output+256)/512)*512 clamp
  [1024, 32768]; image_tokens = sum(ceil(w/28)*ceil(h/28)/4) per frame (need
  frame dims: caller passes list[(w,h)]).
"""
