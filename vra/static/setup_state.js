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
    const engineApiKey = fields.engineApiKey.value.trim();
    const asrBase = fields.asrBase.value.trim();
    const lines = [
      "# Video RSS Aggregator generated configuration",
      `BIND_ADDRESS=${fields.bind.value.trim()}`,
      `VRA_DATA_DIR=${fields.storage.value.trim()}`,
      `VRA_DATABASE_PATH=${fields.db.value.trim()}`,
      `VRA_ENGINE_BASE_URL=${fields.engine.value.trim()}`,
      `VRA_MODEL=${fields.model.value.trim()}`,
      `VRA_MODEL_CTX=${fields.modelCtx.value.trim()}`,
      `VRA_THINK=${fields.think.value.trim()}`,
      `VRA_ASR=${fields.asr.value.trim()}`,
      `VRA_ASR_MODEL=${fields.asrModel.value.trim()}`,
      `VRA_MAX_FRAMES=${fields.maxFrames.value.trim()}`,
      `VRA_FRAME_MODE=${fields.frameMode.value.trim()}`,
      `VRA_SCENE_THRESHOLD=${fields.sceneThreshold.value.trim()}`,
      `VRA_FRAME_MAX_DIM=${fields.frameMaxDim.value.trim()}`,
      `VRA_MAX_TRANSCRIPT_CHARS=${fields.maxTranscript.value.trim()}`,
      `VRA_MAX_OUTPUT_TOKENS=${fields.outputTokens.value.trim()}`,
      `VRA_POLL_MINUTES=${fields.pollMinutes.value.trim()}`,
      `VRA_CPU_WORKERS=${fields.cpuWorkers.value.trim()}`,
      `VRA_RSS_TITLE=${fields.rssTitle.value.trim()}`,
      `VRA_RSS_LINK=${fields.rssLink.value.trim()}`,
      `VRA_RSS_DESCRIPTION=${fields.rssDescription.value.trim()}`,
    ];

    if (apiKey) {
      lines.splice(2, 0, `API_KEY=${apiKey}`);
    }
    if (engineApiKey) {
      lines.splice(3, 0, `VRA_ENGINE_API_KEY=${engineApiKey}`);
    }
    if (asrBase) {
      lines.push(`VRA_ASR_BASE_URL=${asrBase}`);
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
    fields.apiKey.value = "";
    fields.storage.value = defaults.data_dir || ".data";
    fields.db.value = defaults.database_path || ".data/vra.db";
    fields.engine.value = defaults.engine_base_url || "http://127.0.0.1:8000/v1";
    fields.engineApiKey.value = "";
    fields.model.value = defaults.model || "qwen3.5-9b";
    fields.modelCtx.value = defaults.model_ctx ?? 32768;
    fields.think.value = defaults.think || "off";
    fields.asr.value = defaults.asr || "auto";
    fields.asrBase.value = defaults.asr_base_url || "";
    fields.asrModel.value = defaults.asr_model || "Qwen3-ASR-1.7B";
    fields.maxFrames.value = defaults.max_frames ?? 8;
    fields.frameMode.value = defaults.frame_mode || "auto";
    fields.sceneThreshold.value = defaults.scene_threshold ?? 0.3;
    fields.frameMaxDim.value = defaults.frame_max_dim ?? 768;
    fields.maxTranscript.value = defaults.max_transcript_chars ?? 24000;
    fields.outputTokens.value = defaults.max_output_tokens ?? 1024;
    fields.pollMinutes.value = defaults.poll_minutes ?? 15;
    fields.cpuWorkers.value = defaults.cpu_workers ?? 2;
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
