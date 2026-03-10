/* ─── Micro-interaction layer ───────────────────────────────── */
/* Preserves all existing business logic; adds reveal, press   */
/* feedback, toast, shimmer, and smooth-scroll behaviours.     */

const defaults = JSON.parse(document.getElementById("setup-defaults").textContent);

const fields = {
  bind: document.getElementById("bind"),
  apiKey: document.getElementById("api-key"),
  storage: document.getElementById("storage"),
  db: document.getElementById("db"),
  ollama: document.getElementById("ollama"),
  modelPrimary: document.getElementById("model-primary"),
  modelFallback: document.getElementById("model-fallback"),
  modelMin: document.getElementById("model-min"),
  autoPullModels: document.getElementById("auto-pull-models"),
  budget: document.getElementById("budget"),
  budgetRatio: document.getElementById("budget-ratio"),
  reserve: document.getElementById("reserve"),
  contextTokens: document.getElementById("context-tokens"),
  outputTokens: document.getElementById("output-tokens"),
  maxFrames: document.getElementById("max-frames"),
  sceneDetection: document.getElementById("scene-detection"),
  sceneThreshold: document.getElementById("scene-threshold"),
  sceneMin: document.getElementById("scene-min"),
  maxTranscript: document.getElementById("max-transcript"),
  transcriptRetention: document.getElementById("transcript-retention"),
  summaryRetention: document.getElementById("summary-retention"),
  rssTitle: document.getElementById("rss-title"),
  rssLink: document.getElementById("rss-link"),
  rssDescription: document.getElementById("rss-description"),
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

/* ─── Toast ─────────────────────────────────────────────────── */
const toastEl = document.getElementById("toast");
let toastTimer = null;

function showToast(message, duration = 2200) {
  toastEl.textContent = message;
  toastEl.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.classList.remove("visible");
  }, duration);
}

/* ─── Status helpers ────────────────────────────────────────── */
function setStatus(target, message, isError = false) {
  target.style.opacity = "0";
  requestAnimationFrame(() => {
    target.textContent = message;
    target.classList.toggle("error", isError);
    target.style.opacity = "";
    target.classList.remove("fade-in");
    void target.offsetWidth; /* force reflow for re-trigger */
    target.classList.add("fade-in");
  });
}

function asBool(value) {
  return value ? "true" : "false";
}

/* ─── Shimmer helpers ───────────────────────────────────────── */
function startLoading(preEl) {
  preEl.classList.add("loading");
}

function stopLoading(preEl) {
  preEl.classList.remove("loading");
}

/* ─── Smooth scroll to element ──────────────────────────────── */
function scrollIntoViewSmooth(el) {
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ─── Env block renderer ────────────────────────────────────── */
function renderEnvBlock() {
  const apiKey = fields.apiKey.value.trim();
  const lines = [
    "# Video RSS Aggregator generated configuration",
    `BIND_ADDRESS=${fields.bind.value.trim()}`,
    `VRA_STORAGE_DIR=${fields.storage.value.trim()}`,
    `VRA_DATABASE_PATH=${fields.db.value.trim()}`,
    `VRA_OLLAMA_BASE_URL=${fields.ollama.value.trim()}`,
    `VRA_MODEL_PRIMARY=${fields.modelPrimary.value.trim()}`,
    `VRA_MODEL_FALLBACK=${fields.modelFallback.value.trim()}`,
    `VRA_MODEL_MIN=${fields.modelMin.value.trim()}`,
    `VRA_AUTO_PULL_MODELS=${asBool(fields.autoPullModels.checked)}`,
    `VRA_VRAM_BUDGET_MB=${fields.budget.value.trim()}`,
    `VRA_MODEL_SIZE_BUDGET_RATIO=${fields.budgetRatio.value.trim()}`,
    `VRA_MODEL_SELECTION_RESERVE_MB=${fields.reserve.value.trim()}`,
    `VRA_CONTEXT_TOKENS=${fields.contextTokens.value.trim()}`,
    `VRA_MAX_OUTPUT_TOKENS=${fields.outputTokens.value.trim()}`,
    `VRA_MAX_FRAMES=${fields.maxFrames.value.trim()}`,
    `VRA_FRAME_SCENE_DETECTION=${asBool(fields.sceneDetection.checked)}`,
    `VRA_FRAME_SCENE_THRESHOLD=${fields.sceneThreshold.value.trim()}`,
    `VRA_FRAME_SCENE_MIN_FRAMES=${fields.sceneMin.value.trim()}`,
    `VRA_MAX_TRANSCRIPT_CHARS=${fields.maxTranscript.value.trim()}`,
    `VRA_TRANSCRIPT_RETENTION_PER_VIDEO=${fields.transcriptRetention.value.trim()}`,
    `VRA_SUMMARY_RETENTION_PER_VIDEO=${fields.summaryRetention.value.trim()}`,
    `VRA_RSS_TITLE=${fields.rssTitle.value.trim()}`,
    `VRA_RSS_LINK=${fields.rssLink.value.trim()}`,
    `VRA_RSS_DESCRIPTION=${fields.rssDescription.value.trim()}`,
  ];
  if (apiKey) {
    lines.splice(2, 0, `API_KEY=${apiKey}`);
  }
  outputs.env.textContent = lines.join("\n");
}

async function copyEnvBlock() {
  try {
    await navigator.clipboard.writeText(outputs.env.textContent);
    showToast("Copied to clipboard");
  } catch (error) {
    setStatus(status.env, `Copy failed: ${error}`, true);
  }
}

/* ─── API helpers ───────────────────────────────────────────── */
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

/* ─── API operations ────────────────────────────────────────── */
async function loadSetupConfig() {
  try {
    const configData = await apiFetch("setup/config");
    outputs.runtime.textContent = JSON.stringify(configData, null, 2);
    setStatus(status.runtime, "Loaded setup defaults.");
  } catch (error) {
    setStatus(status.runtime, `Unable to load setup config: ${error}`, true);
  }
}

async function runDiagnostics() {
  setStatus(status.diagnostics, "Running diagnostics...");
  startLoading(outputs.diagnostics);
  try {
    const report = await apiFetch("setup/diagnostics");
    outputs.diagnostics.textContent = JSON.stringify(report, null, 2);
    if (report.ready) {
      setStatus(status.diagnostics, "All required dependencies are available.");
    } else {
      setStatus(status.diagnostics, "Some dependencies are missing or unreachable.", true);
    }
  } catch (error) {
    setStatus(status.diagnostics, `Diagnostics failed: ${error}`, true);
  } finally {
    stopLoading(outputs.diagnostics);
    scrollIntoViewSmooth(outputs.diagnostics);
  }
}

async function checkRuntime() {
  setStatus(status.runtime, "Checking runtime...");
  startLoading(outputs.runtime);
  try {
    const runtime = await apiFetch("runtime");
    outputs.runtime.textContent = JSON.stringify(runtime, null, 2);
    setStatus(status.runtime, "Runtime check completed.");
  } catch (error) {
    setStatus(status.runtime, `Runtime check failed: ${error}`, true);
  } finally {
    stopLoading(outputs.runtime);
    scrollIntoViewSmooth(outputs.runtime);
  }
}

async function bootstrapModels() {
  setStatus(status.runtime, "Bootstrapping models...");
  startLoading(outputs.runtime);
  try {
    const report = await apiFetch("setup/bootstrap", { method: "POST" });
    outputs.runtime.textContent = JSON.stringify(report, null, 2);
    setStatus(status.runtime, "Models are ready.");
  } catch (error) {
    setStatus(status.runtime, `Bootstrap failed: ${error}`, true);
  } finally {
    stopLoading(outputs.runtime);
    scrollIntoViewSmooth(outputs.runtime);
  }
}

async function runProcess() {
  const sourceUrl = fields.processSource.value.trim();
  if (!sourceUrl) {
    setStatus(status.process, "Please provide a source URL or local path.", true);
    return;
  }

  setStatus(status.process, "Processing source... this may take a while.");
  startLoading(outputs.process);
  try {
    const payload = {
      source_url: sourceUrl,
      title: fields.processTitle.value.trim() || null,
    };
    const result = await apiFetch("process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    outputs.process.textContent = JSON.stringify(result, null, 2);
    setStatus(status.process, "Processing completed.");
  } catch (error) {
    setStatus(status.process, `Process request failed: ${error}`, true);
  } finally {
    stopLoading(outputs.process);
    scrollIntoViewSmooth(outputs.process);
  }
}

/* ─── Hydrate defaults ──────────────────────────────────────── */
function hydrateDefaults() {
  fields.bind.value = defaults.bind_address || "127.0.0.1:8080";
  fields.apiKey.value = defaults.api_key || "";
  fields.storage.value = defaults.storage_dir || ".data";
  fields.db.value = defaults.database_path || ".data/vra.db";
  fields.ollama.value = defaults.ollama_base_url || "http://127.0.0.1:11434";
  fields.modelPrimary.value = defaults.model_primary || "qwen3.5:4b-q4_K_M";
  fields.modelFallback.value = defaults.model_fallback || "qwen3.5:2b-q4_K_M";
  fields.modelMin.value = defaults.model_min || "qwen3.5:0.8b-q8_0";
  fields.autoPullModels.checked = Boolean(defaults.auto_pull_models);
  fields.budget.value = defaults.vram_budget_mb ?? 8192;
  fields.budgetRatio.value = defaults.model_size_budget_ratio ?? 0.75;
  fields.reserve.value = defaults.model_selection_reserve_mb ?? 768;
  fields.contextTokens.value = defaults.context_tokens ?? 3072;
  fields.outputTokens.value = defaults.max_output_tokens ?? 768;
  fields.maxFrames.value = defaults.max_frames ?? 5;
  fields.sceneDetection.checked = Boolean(defaults.frame_scene_detection);
  fields.sceneThreshold.value = defaults.frame_scene_threshold ?? 0.28;
  fields.sceneMin.value = defaults.frame_scene_min_frames ?? 2;
  fields.maxTranscript.value = defaults.max_transcript_chars ?? 16000;
  fields.transcriptRetention.value = defaults.transcript_retention_per_video ?? 3;
  fields.summaryRetention.value = defaults.summary_retention_per_video ?? 5;
  fields.rssTitle.value = defaults.rss_title || "Video RSS Aggregator";
  fields.rssLink.value = defaults.rss_link || "http://127.0.0.1:8080/rss";
  fields.rssDescription.value = defaults.rss_description || "Video summaries";
  fields.processSource.value = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
  renderEnvBlock();
}

/* ─── Button press feedback ─────────────────────────────────── */
function addPressEffect(button) {
  button.addEventListener("mousedown", () => {
    button.classList.remove("pressing");
    void button.offsetWidth;
    button.classList.add("pressing");
  });

  button.addEventListener("animationend", () => {
    button.classList.remove("pressing");
  });
}

document.querySelectorAll("button").forEach(addPressEffect);

/* ─── Event bindings ────────────────────────────────────────── */
document.getElementById("copy-env").addEventListener("click", copyEnvBlock);
document.getElementById("run-diagnostics").addEventListener("click", runDiagnostics);
document.getElementById("runtime-check").addEventListener("click", checkRuntime);
document.getElementById("bootstrap-models").addEventListener("click", bootstrapModels);
document.getElementById("process-run").addEventListener("click", runProcess);

Object.values(fields).forEach((node) => {
  if (!node || node === fields.processSource || node === fields.processTitle) {
    return;
  }
  node.addEventListener("input", renderEnvBlock);
  node.addEventListener("change", renderEnvBlock);
});

/* ─── Intersection Observer — staggered card reveal ─────────── */
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const card = entry.target;
        const delay = getComputedStyle(card).getPropertyValue("--reveal-delay").trim();
        const ms = parseInt(delay, 10) || 0;
        setTimeout(() => {
          card.classList.add("revealed");
        }, ms);
        revealObserver.unobserve(card);
      }
    });
  },
  { threshold: 0.08 }
);

document.querySelectorAll(".card").forEach((card) => {
  revealObserver.observe(card);
});

/* ─── Boot ──────────────────────────────────────────────────── */
hydrateDefaults();
runDiagnostics();
loadSetupConfig();
