"""FastAPI application. OWNER: agent-D. create_app(runtime) per docs/v2/SPEC.md API table.

- Lifespan builds AppRuntime via vra.runtime (composition root, lead-owned:
  runtime.build_runtime(config) -> object with .use_cases attrs; agent-D codes
  against runtime facade attributes documented in vra/runtime.py docstring).
- Endpoints exactly as SPEC; auth dependency (API_KEY bearer/x-api-key) on
  mutating routes; /rss + /health + /videos open.
- Static: mount vra/static + serve extracted frame files at
  /media/{video_id}/{idx}.jpg from data dir (path-traversal safe: video_id/idx
  validated ^[0-9a-f-]{36}$ and ^\\d+$).
- Request body size < 1MB; /setup endpoints port v1 behavior using new config.
"""
