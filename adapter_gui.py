from __future__ import annotations

import json

from core_config import Config


def _setup_defaults(config: Config) -> dict[str, object]:
    return {
        "bind_address": f"{config.bind_host}:{config.bind_port}",
        "storage_dir": config.storage_dir,
        "database_path": config.database_path,
        "ollama_base_url": config.ollama_base_url,
        "vram_budget_mb": config.vram_budget_mb,
        "model_selection_reserve_mb": config.model_selection_reserve_mb,
        "max_frames": config.max_frames,
        "frame_scene_detection": config.frame_scene_detection,
        "frame_scene_threshold": config.frame_scene_threshold,
        "frame_scene_min_frames": config.frame_scene_min_frames,
        "model_primary": config.model_primary,
        "model_fallback": config.model_fallback,
        "model_min": config.model_min,
    }


def render_setup_page(config: Config) -> str:
    defaults_json = json.dumps(_setup_defaults(config)).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Video RSS Aggregator Studio</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
      --bg-deep: #071723;
      --bg-mid: #11314a;
      --bg-soft: #1b4f72;
      --panel: rgba(8, 30, 46, 0.76);
      --panel-border: rgba(97, 171, 201, 0.35);
      --text-main: #ecf7ff;
      --text-muted: #a5c3d6;
      --accent: #1ec7a4;
      --accent-warm: #ff9a56;
      --danger: #ff6b6b;
      --radius: 18px;
      --shadow: 0 20px 60px rgba(3, 12, 20, 0.55);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Sora", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text-main);
      background:
        radial-gradient(circle at 8% 15%, rgba(30, 199, 164, 0.2), transparent 40%),
        radial-gradient(circle at 85% 8%, rgba(255, 154, 86, 0.18), transparent 44%),
        linear-gradient(140deg, var(--bg-deep), var(--bg-mid) 55%, var(--bg-soft));
      min-height: 100vh;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: radial-gradient(circle at center, black 45%, transparent 100%);
    }

    .shell {
      width: min(1180px, 92vw);
      margin: 0 auto;
      padding: 34px 0 42px;
      position: relative;
      z-index: 1;
      animation: entry 0.65s ease-out;
    }

    .hero {
      margin-bottom: 22px;
      padding: 28px;
      border-radius: var(--radius);
      background: linear-gradient(120deg, rgba(13, 40, 61, 0.82), rgba(10, 27, 41, 0.78));
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
    }

    .eyebrow {
      margin: 0;
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }

    h1 {
      margin: 10px 0 8px;
      font-size: clamp(1.75rem, 2.8vw, 2.4rem);
      line-height: 1.1;
    }

    .subhead {
      margin: 0;
      color: var(--text-muted);
      max-width: 780px;
      line-height: 1.6;
    }

    .grid {
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }

    .card {
      grid-column: span 6;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      animation: rise 0.55s ease-out;
    }

    .card:nth-child(2) { animation-delay: 0.06s; }
    .card:nth-child(3) { animation-delay: 0.12s; }
    .card:nth-child(4) { animation-delay: 0.18s; }

    .card h2 {
      margin: 0 0 10px;
      font-size: 1.05rem;
    }

    .muted {
      margin: 0 0 12px;
      color: var(--text-muted);
      line-height: 1.5;
      font-size: 0.93rem;
    }

    ol {
      margin: 0;
      padding-left: 18px;
      line-height: 1.6;
      color: var(--text-muted);
    }

    code,
    pre {
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 0.84rem;
    }

    .field-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr 1fr;
      margin-bottom: 12px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .field label {
      color: var(--text-muted);
      font-size: 0.82rem;
      letter-spacing: 0.02em;
    }

    input[type="text"],
    input[type="number"],
    textarea {
      width: 100%;
      border: 1px solid rgba(150, 206, 230, 0.35);
      border-radius: 12px;
      padding: 10px 11px;
      background: rgba(3, 18, 28, 0.75);
      color: var(--text-main);
      outline: none;
      transition: border-color 0.2s ease, transform 0.2s ease;
    }

    input:focus,
    textarea:focus {
      border-color: var(--accent);
      transform: translateY(-1px);
    }

    .inline {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 8px 0 12px;
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 10px;
    }

    button {
      border: none;
      border-radius: 999px;
      padding: 10px 15px;
      color: #062135;
      font-weight: 700;
      cursor: pointer;
      background: linear-gradient(130deg, var(--accent), #72f2d8);
      transition: transform 0.2s ease, filter 0.2s ease;
    }

    button.warm {
      background: linear-gradient(130deg, var(--accent-warm), #ffc48d);
      color: #2a1300;
    }

    button.ghost {
      background: rgba(130, 187, 213, 0.16);
      color: var(--text-main);
      border: 1px solid rgba(150, 206, 230, 0.3);
    }

    button:hover {
      transform: translateY(-1px);
      filter: saturate(1.05);
    }

    pre {
      margin: 0;
      border: 1px solid rgba(150, 206, 230, 0.3);
      border-radius: 12px;
      background: rgba(2, 16, 26, 0.8);
      padding: 12px;
      min-height: 136px;
      max-height: 280px;
      overflow: auto;
      line-height: 1.45;
      color: #d8efff;
    }

    .status {
      margin-top: 10px;
      font-size: 0.86rem;
      color: var(--text-muted);
    }

    .status.error {
      color: var(--danger);
    }

    @media (max-width: 980px) {
      .card,
      .card.wide {
        grid-column: span 12;
      }

      .field-grid {
        grid-template-columns: 1fr;
      }
    }

    @keyframes entry {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(18px) scale(0.99); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">Windows-native setup studio</p>
      <h1>Video RSS Aggregator Installation + Configuration</h1>
      <p class="subhead">
        This workspace helps you install dependencies, generate your environment configuration,
        bootstrap local Qwen 3.5 models in Ollama, and run your first source processing request.
      </p>
    </section>

    <section class="grid">
      <article class="card">
        <h2>1) Install Prerequisites</h2>
        <p class="muted">Follow this sequence before starting API operations.</p>
        <ol>
          <li>Install Python 3.11+ and make sure <code>python</code> is on <code>PATH</code>.</li>
          <li>Install Ollama for Windows and open it once so the API is online.</li>
          <li>Install FFmpeg/FFprobe and confirm both commands resolve in terminal.</li>
          <li>In this repo run <code>pip install -e .</code>, then <code>python -m vra bootstrap</code>.</li>
          <li>Run <code>python -m vra serve --bind 127.0.0.1:8080</code> to keep this GUI active.</li>
        </ol>
        <div class="actions" style="margin-top: 12px;">
          <button id="run-diagnostics" class="ghost" type="button">Run dependency diagnostics</button>
        </div>
        <pre id="diagnostics-output"></pre>
        <p class="status" id="diagnostics-status"></p>
      </article>

      <article class="card">
        <h2>2) Configuration Builder</h2>
        <p class="muted">Tune settings and copy the generated <code>.env</code> block. Restart server after edits.</p>
        <div class="field-grid">
          <div class="field">
            <label for="bind">Bind address</label>
            <input id="bind" type="text" />
          </div>
          <div class="field">
            <label for="ollama">Ollama API URL</label>
            <input id="ollama" type="text" />
          </div>
          <div class="field">
            <label for="storage">Storage directory</label>
            <input id="storage" type="text" />
          </div>
          <div class="field">
            <label for="db">Database path</label>
            <input id="db" type="text" />
          </div>
          <div class="field">
            <label for="budget">VRAM budget (MB)</label>
            <input id="budget" type="number" min="1024" step="128" />
          </div>
          <div class="field">
            <label for="reserve">Selection reserve (MB)</label>
            <input id="reserve" type="number" min="0" step="64" />
          </div>
          <div class="field">
            <label for="frames">Max frames</label>
            <input id="frames" type="number" min="1" max="16" step="1" />
          </div>
          <div class="field">
            <label for="scene-threshold">Scene threshold</label>
            <input id="scene-threshold" type="number" min="0.05" max="0.95" step="0.01" />
          </div>
          <div class="field">
            <label for="scene-min">Min scene frames</label>
            <input id="scene-min" type="number" min="1" max="16" step="1" />
          </div>
          <div class="field">
            <label for="model-primary">Primary model</label>
            <input id="model-primary" type="text" />
          </div>
        </div>
        <label class="inline">
          <input id="scene-detection" type="checkbox" />
          Enable scene-aware frame extraction
        </label>
        <div class="actions">
          <button id="copy-env" class="ghost" type="button">Copy .env block</button>
        </div>
        <pre id="env-output"></pre>
        <p class="status" id="env-status"></p>
      </article>

      <article class="card">
        <h2>3) Runtime Validation</h2>
        <p class="muted">Use API key below only if your server requires auth.</p>
        <div class="field">
          <label for="api-key">API key (optional)</label>
          <input id="api-key" type="text" placeholder="Leave blank when auth is disabled" />
        </div>
        <div class="actions">
          <button id="runtime-check" type="button">Check runtime</button>
          <button id="bootstrap-models" class="warm" type="button">Bootstrap models</button>
        </div>
        <pre id="runtime-output"></pre>
        <p class="status" id="runtime-status"></p>
      </article>

      <article class="card">
        <h2>4) First Processing Run</h2>
        <p class="muted">Process a URL or local media path and inspect summary outputs.</p>
        <div class="field-grid">
          <div class="field" style="grid-column: span 2;">
            <label for="process-source">Source URL or file path</label>
            <input id="process-source" type="text" placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ" />
          </div>
          <div class="field" style="grid-column: span 2;">
            <label for="process-title">Title override (optional)</label>
            <input id="process-title" type="text" />
          </div>
        </div>
        <div class="actions">
          <button id="process-run" type="button">Run processing</button>
        </div>
        <pre id="process-output"></pre>
        <p class="status" id="process-status"></p>
      </article>
    </section>
  </main>

  <script id="setup-defaults" type="application/json">__DEFAULTS_JSON__</script>
  <script>
    const defaults = JSON.parse(document.getElementById("setup-defaults").textContent);

    const fields = {
      bind: document.getElementById("bind"),
      ollama: document.getElementById("ollama"),
      storage: document.getElementById("storage"),
      db: document.getElementById("db"),
      budget: document.getElementById("budget"),
      reserve: document.getElementById("reserve"),
      frames: document.getElementById("frames"),
      sceneDetection: document.getElementById("scene-detection"),
      sceneThreshold: document.getElementById("scene-threshold"),
      sceneMin: document.getElementById("scene-min"),
      modelPrimary: document.getElementById("model-primary"),
      apiKey: document.getElementById("api-key"),
      processSource: document.getElementById("process-source"),
      processTitle: document.getElementById("process-title"),
    };

    const outputs = {
      diagnostics: document.getElementById("diagnostics-output"),
      env: document.getElementById("env-output"),
      runtime: document.getElementById("runtime-output"),
      process: document.getElementById("process-output"),
    };

    const status = {
      diagnostics: document.getElementById("diagnostics-status"),
      env: document.getElementById("env-status"),
      runtime: document.getElementById("runtime-status"),
      process: document.getElementById("process-status"),
    };

    function setStatus(target, message, isError = false) {
      target.textContent = message;
      target.classList.toggle("error", isError);
    }

    function asBool(value) {
      return value ? "true" : "false";
    }

    function renderEnvBlock() {
      const lines = [
        "# Video RSS Aggregator generated configuration",
        `BIND_ADDRESS=${fields.bind.value.trim()}`,
        `VRA_OLLAMA_BASE_URL=${fields.ollama.value.trim()}`,
        `VRA_STORAGE_DIR=${fields.storage.value.trim()}`,
        `VRA_DATABASE_PATH=${fields.db.value.trim()}`,
        `VRA_VRAM_BUDGET_MB=${fields.budget.value.trim()}`,
        `VRA_MODEL_SELECTION_RESERVE_MB=${fields.reserve.value.trim()}`,
        `VRA_MAX_FRAMES=${fields.frames.value.trim()}`,
        `VRA_FRAME_SCENE_DETECTION=${asBool(fields.sceneDetection.checked)}`,
        `VRA_FRAME_SCENE_THRESHOLD=${fields.sceneThreshold.value.trim()}`,
        `VRA_FRAME_SCENE_MIN_FRAMES=${fields.sceneMin.value.trim()}`,
        `VRA_MODEL_PRIMARY=${fields.modelPrimary.value.trim()}`,
      ];
      outputs.env.textContent = lines.join("\n");
    }

    async function copyEnvBlock() {
      try {
        await navigator.clipboard.writeText(outputs.env.textContent);
        setStatus(status.env, "Copied .env block to clipboard.");
      } catch (error) {
        setStatus(status.env, `Copy failed: ${error}`, true);
      }
    }

    function authHeaders() {
      const token = fields.apiKey.value.trim();
      if (!token) {
        return {};
      }
      return { "X-API-Key": token };
    }

    async function apiFetch(path, options = {}) {
      const headers = {
        ...(options.headers || {}),
        ...authHeaders(),
      };

      const response = await fetch(path, {
        ...options,
        headers,
      });

      const bodyText = await response.text();
      let data;
      try {
        data = JSON.parse(bodyText);
      } catch {
        data = bodyText;
      }

      if (!response.ok) {
        throw new Error(typeof data === "string" ? data : JSON.stringify(data, null, 2));
      }
      return data;
    }

    async function loadSetupConfig() {
      try {
        const configData = await apiFetch("/setup/config");
        outputs.runtime.textContent = JSON.stringify(configData, null, 2);
        setStatus(status.runtime, "Loaded setup defaults.");
      } catch (error) {
        setStatus(status.runtime, `Unable to load setup config: ${error}`, true);
      }
    }

    async function runDiagnostics() {
      setStatus(status.diagnostics, "Running diagnostics...");
      try {
        const report = await apiFetch("/setup/diagnostics");
        outputs.diagnostics.textContent = JSON.stringify(report, null, 2);
        if (report.ready) {
          setStatus(status.diagnostics, "All required dependencies are available.");
        } else {
          setStatus(status.diagnostics, "Some dependencies are missing or unreachable.", true);
        }
      } catch (error) {
        setStatus(status.diagnostics, `Diagnostics failed: ${error}`, true);
      }
    }

    async function checkRuntime() {
      setStatus(status.runtime, "Checking runtime...");
      try {
        const runtime = await apiFetch("/runtime");
        outputs.runtime.textContent = JSON.stringify(runtime, null, 2);
        setStatus(status.runtime, "Runtime check completed.");
      } catch (error) {
        setStatus(status.runtime, `Runtime check failed: ${error}`, true);
      }
    }

    async function bootstrapModels() {
      setStatus(status.runtime, "Bootstrapping models...");
      try {
        const report = await apiFetch("/setup/bootstrap", { method: "POST" });
        outputs.runtime.textContent = JSON.stringify(report, null, 2);
        setStatus(status.runtime, "Models are ready.");
      } catch (error) {
        setStatus(status.runtime, `Bootstrap failed: ${error}`, true);
      }
    }

    async function runProcess() {
      const sourceUrl = fields.processSource.value.trim();
      if (!sourceUrl) {
        setStatus(status.process, "Please provide a source URL or local path.", true);
        return;
      }

      setStatus(status.process, "Processing source... this may take a while.");
      try {
        const payload = {
          source_url: sourceUrl,
          title: fields.processTitle.value.trim() || null,
        };
        const result = await apiFetch("/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        outputs.process.textContent = JSON.stringify(result, null, 2);
        setStatus(status.process, "Processing completed.");
      } catch (error) {
        setStatus(status.process, `Process request failed: ${error}`, true);
      }
    }

    function hydrateDefaults() {
      fields.bind.value = defaults.bind_address || "127.0.0.1:8080";
      fields.ollama.value = defaults.ollama_base_url || "http://127.0.0.1:11434";
      fields.storage.value = defaults.storage_dir || ".data";
      fields.db.value = defaults.database_path || ".data/vra.db";
      fields.budget.value = defaults.vram_budget_mb ?? 8192;
      fields.reserve.value = defaults.model_selection_reserve_mb ?? 768;
      fields.frames.value = defaults.max_frames ?? 5;
      fields.sceneDetection.checked = Boolean(defaults.frame_scene_detection);
      fields.sceneThreshold.value = defaults.frame_scene_threshold ?? 0.28;
      fields.sceneMin.value = defaults.frame_scene_min_frames ?? 2;
      fields.modelPrimary.value = defaults.model_primary || "qwen3.5:4b-q4_K_M";
      fields.processSource.value = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
      renderEnvBlock();
    }

    document.getElementById("copy-env").addEventListener("click", copyEnvBlock);
    document.getElementById("run-diagnostics").addEventListener("click", runDiagnostics);
    document.getElementById("runtime-check").addEventListener("click", checkRuntime);
    document.getElementById("bootstrap-models").addEventListener("click", bootstrapModels);
    document.getElementById("process-run").addEventListener("click", runProcess);

    [
      fields.bind,
      fields.ollama,
      fields.storage,
      fields.db,
      fields.budget,
      fields.reserve,
      fields.frames,
      fields.sceneThreshold,
      fields.sceneMin,
      fields.modelPrimary,
      fields.sceneDetection,
    ].forEach((node) => {
      node.addEventListener("input", renderEnvBlock);
      node.addEventListener("change", renderEnvBlock);
    });

    hydrateDefaults();
    runDiagnostics();
    loadSetupConfig();
  </script>
</body>
</html>
""".replace("__DEFAULTS_JSON__", defaults_json)
