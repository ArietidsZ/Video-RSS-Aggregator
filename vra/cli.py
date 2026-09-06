"""Click CLI. OWNER: agent-D. Commands: serve|ingest|process|status|bootstrap|
benchmark|cleanup per SPEC. Entry point vra=cli:cli stays. serve builds runtime
(vra.runtime.build_runtime) and runs uvicorn with lifespan owning scheduler+workers.
"""
