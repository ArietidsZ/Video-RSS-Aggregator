function asBool(value) {
  return value ? "true" : "false";
}

const STEP_ORDER = ["prerequisites", "configuration", "runtime", "process"];

export function createSetupState({ defaults = {}, fields = {} } = {}) {
  const steps = {
    prerequisites: { state: "unverified" },
    configuration: { state: "ready" },
    runtime: { state: "unverified" },
    process: { state: "unverified", stale: false, summary: null, error: null },
  };
  const shellState = { steps };

  function getStepStatus(stepId) {
    if (stepId === "configuration" && steps.prerequisites.state !== "complete") {
      return "blocked";
    }

    if (stepId === "runtime" && steps.configuration.state !== "complete") {
      return "blocked";
    }

    if (stepId === "process" && steps.runtime.state !== "complete") {
      return "blocked";
    }

    return steps[stepId].state;
  }

  function getActiveStep() {
    const runningStep = STEP_ORDER.find((stepId) => steps[stepId].state === "running");
    if (runningStep) {
      return runningStep;
    }

    const failedStep = STEP_ORDER.find((stepId) => steps[stepId].state === "failed");
    if (failedStep) {
      return failedStep;
    }

    return getRecommendedNextStep() || "process";
  }

  function getShellState() {
    return {
      activeStep: getActiveStep(),
      steps: STEP_ORDER.map((stepId) => ({
        id: stepId,
        status: getStepStatus(stepId),
      })),
    };
  }

  function buildEnvLines() {
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

    return lines;
  }

  function authHeaders() {
    const token = fields.apiKey.value.trim();
    return token ? { "X-API-Key": token } : {};
  }

  function beginDiagnosticsCheck() {
    steps.prerequisites.state = "running";
  }

  function applyDiagnosticsResult(view) {
    const diagnosticsReady = view.state === "ready";

    steps.prerequisites.state = diagnosticsReady ? "complete" : "blocked";
    steps.configuration.state = "ready";
    steps.runtime.state = "unverified";
    steps.process = { ...steps.process, state: "unverified", error: null };
  }

  function markConfigurationComplete() {
    if (steps.prerequisites.state !== "complete") {
      return false;
    }

    steps.configuration.state = "complete";
    return true;
  }

  function beginRuntimeCheck() {
    if (steps.prerequisites.state !== "complete") {
      return false;
    }

    if (steps.configuration.state !== "complete") {
      return false;
    }

    steps.runtime.state = "running";
    return true;
  }

  function completeRuntimeCheck(view) {
    const runtimeReady = view.state === "ready";

    steps.runtime.state = runtimeReady ? "complete" : "blocked";
    steps.process = {
      ...steps.process,
      state: "unverified",
      error: runtimeReady ? null : steps.process.error,
    };
  }

  function beginProcess() {
    if (getStepStatus("runtime") !== "complete") {
      return false;
    }

    steps.process.state = "running";
    return true;
  }

  function markProcessSuccess(summary) {
    steps.process = { state: "complete", stale: false, summary, error: null };
  }

  function markProcessFailure(error) {
    steps.process = {
      ...steps.process,
      state: "failed",
      stale: Boolean(steps.process.summary),
      error,
    };
  }

  function getRecommendedNextStep() {
    return STEP_ORDER.find((stepId) =>
      ["ready", "unverified", "blocked"].includes(getStepStatus(stepId)),
    );
  }

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
  }

  return {
    authHeaders,
    beginDiagnosticsCheck,
    beginProcess,
    beginRuntimeCheck,
    buildEnvLines,
    applyDiagnosticsResult,
    completeRuntimeCheck,
    getShellState,
    getRecommendedNextStep,
    hydrateDefaults,
    markConfigurationComplete,
    markProcessFailure,
    markProcessSuccess,
    shellState,
    steps,
  };
}
