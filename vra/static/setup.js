import { createSetupApi } from "./setup_api.js";
import { createSetupState } from "./setup_state.js";
import {
  buildProcessFailureView,
  buildShellSummaryView,
  buildVideoDetailSummaryView,
} from "./setup_view_models.js";
import { createSetupViews } from "./setup_views.js";

const defaults = JSON.parse(document.getElementById("setup-defaults").textContent);

const fields = {
  bind: document.getElementById("bind"),
  apiKey: document.getElementById("api-key"),
  storage: document.getElementById("storage"),
  db: document.getElementById("db"),
  engine: document.getElementById("engine"),
  engineApiKey: document.getElementById("engine-api-key"),
  model: document.getElementById("model"),
  modelCtx: document.getElementById("model-ctx"),
  think: document.getElementById("think"),
  asr: document.getElementById("asr"),
  asrBase: document.getElementById("asr-base"),
  asrModel: document.getElementById("asr-model"),
  maxFrames: document.getElementById("max-frames"),
  frameMode: document.getElementById("frame-mode"),
  sceneThreshold: document.getElementById("scene-threshold"),
  frameMaxDim: document.getElementById("frame-max-dim"),
  maxTranscript: document.getElementById("max-transcript"),
  outputTokens: document.getElementById("output-tokens"),
  pollMinutes: document.getElementById("poll-minutes"),
  cpuWorkers: document.getElementById("cpu-workers"),
  rssTitle: document.getElementById("rss-title"),
  rssLink: document.getElementById("rss-link"),
  rssDescription: document.getElementById("rss-description"),
  processSource: document.getElementById("process-source"),
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

const state = createSetupState({ defaults, fields });
const views = createSetupViews({
  document,
  outputs,
  status,
  toastEl: document.getElementById("toast"),
});
const api = createSetupApi({
  fetchImpl: (...args) => fetch(...args),
  getAuthHeaders: state.authHeaders,
});

function renderShell() {
  const shellState = state.getShellState();
  const shellSummaryView = buildShellSummaryView(shellState);

  views.renderShellState(shellState, shellSummaryView);
}

function toDiagnosticsView(report) {
  return report.setup_view || { state: report.ready ? "ready" : "blocked" };
}

function toRuntimeView(diagnosticsReport) {
  return (
    diagnosticsReport.runtime_view || {
      state: diagnosticsReport.ready ? "ready" : "blocked",
    }
  );
}

function refreshEnvBlock() {
  views.renderEnvBlock(state.buildEnvLines());
}

function renderRuntimeState(runtimeView = null, runtime = null) {
  const currentRuntimeView = runtimeView || { state: "unverified" };
  const currentRuntime = runtime || currentRuntimeView;

  views.renderRuntimeSummary(currentRuntimeView, currentRuntime);
  views.renderRuntimeDetails(currentRuntime);
  views.renderCommonFixes(currentRuntimeView);
}

function renderProcessState(summaryView = null, detailView = null) {
  views.renderProcessSummary(summaryView);
  views.renderProcessDetails(detailView || summaryView || "");
}

function syncConfigurationProgress() {
  refreshEnvBlock();
  const configurationReady = state.markConfigurationComplete();

  renderShell();
  return configurationReady;
}

async function copyEnvBlock() {
  try {
    syncConfigurationProgress();
    await navigator.clipboard.writeText(outputs.env.textContent);
    views.showToast("Copied to clipboard");
  } catch (error) {
    views.setStatus(status.env, `Copy failed: ${error}`, true);
  }
}

async function handleDiagnostics() {
  state.beginDiagnosticsCheck();
  renderShell();
  views.setStatus(status.diagnostics, "Running diagnostics...");
  views.startLoading(outputs.diagnostics);
  try {
    const report = await api.runDiagnostics();
    const diagnosticsView = toDiagnosticsView(report);

    state.applyDiagnosticsResult(diagnosticsView);
    syncConfigurationProgress();
    outputs.diagnostics.textContent = JSON.stringify(report, null, 2);
    views.renderDiagnosticsSummary(diagnosticsView);
    views.renderCommonFixes(diagnosticsView);
    if (diagnosticsView.state === "ready") {
      views.setStatus(status.diagnostics, "All required dependencies are available.");
    } else {
      views.setStatus(status.diagnostics, "Some dependencies are missing or unreachable.", true);
    }
  } catch (error) {
    state.applyDiagnosticsResult({ state: "blocked" });
    views.setStatus(status.diagnostics, `Diagnostics failed: ${error}`, true);
  } finally {
    renderShell();
    views.stopLoading(outputs.diagnostics);
    views.scrollIntoViewSmooth(outputs.diagnostics);
  }
}

async function handleRuntimeCheck() {
  if (!state.beginRuntimeCheck()) {
    renderShell();
    views.setStatus(
      status.runtime,
      "Complete prerequisites and refresh the configuration before running runtime checks.",
      true,
    );
    return;
  }

  renderShell();
  views.setStatus(status.runtime, "Checking runtime...");
  views.startLoading(outputs.runtime);
  try {
    const report = await api.checkRuntime();
    const runtimeView = toRuntimeView(report);

    state.completeRuntimeCheck(runtimeView);
    renderRuntimeState(runtimeView, report.dependencies || report);
    views.setStatus(status.runtime, "Runtime check completed.");
  } catch (error) {
    state.completeRuntimeCheck({ state: "blocked" });
    renderRuntimeState({ state: "blocked" }, { reachable: false, error: String(error) });
    views.setStatus(status.runtime, `Runtime check failed: ${error}`, true);
  } finally {
    renderShell();
    views.stopLoading(outputs.runtime);
    views.scrollIntoViewSmooth(outputs.runtime);
  }
}

async function handleBootstrapModels() {
  if (!state.beginRuntimeCheck()) {
    renderShell();
    views.setStatus(
      status.runtime,
      "Complete prerequisites and refresh the configuration before verifying models.",
      true,
    );
    return;
  }

  renderShell();
  views.setStatus(status.runtime, "Verifying model availability...");
  views.startLoading(outputs.runtime);
  try {
    const report = await api.bootstrapModels();
    const runtimeCheck = await api.checkRuntime();
    const runtimeView = toRuntimeView(runtimeCheck);

    state.completeRuntimeCheck(runtimeView);
    renderRuntimeState(runtimeView, {
      ...(runtimeCheck.dependencies || {}),
      bootstrap: report,
    });
    views.setStatus(status.runtime, "Models are ready.");
  } catch (error) {
    state.completeRuntimeCheck({ state: "blocked" });
    renderRuntimeState({ state: "blocked" }, { reachable: false, error: String(error) });
    views.setStatus(status.runtime, `Verification failed: ${error}`, true);
  } finally {
    renderShell();
    views.stopLoading(outputs.runtime);
    views.scrollIntoViewSmooth(outputs.runtime);
  }
}

async function handleProcessRun() {
  const sourceUrl = fields.processSource.value.trim();
  if (!sourceUrl) {
    views.setStatus(status.process, "Please provide a source URL.", true);
    return;
  }

  if (!state.beginProcess()) {
    renderShell();
    views.setStatus(status.process, "Complete the runtime check before processing a source.", true);
    return;
  }

  renderShell();
  views.setStatus(status.process, "Submitting source... this may take a while.");
  views.startLoading(outputs.process);
  try {
    const job = await api.runProcess({ source_url: sourceUrl, force: false });
    const finished = await api.waitForJob(job.job_id, (current) => {
      views.setStatus(status.process, `Job ${current.state}...`);
    });
    if (finished.state === "failed") {
      throw new Error(finished.error || "processing job failed");
    }
    const detail = await api.getVideo(job.video_id);
    const summaryView = buildVideoDetailSummaryView(detail);
    state.markProcessSuccess(summaryView);
    renderProcessState(summaryView, summaryView);
    views.setStatus(status.process, "Processing completed.");
  } catch (error) {
    state.markProcessFailure(error);
    const priorSummary = state.steps.process.summary
      ? state.steps.process.summary
      : null;
    const failureView = buildProcessFailureView(error, priorSummary);
    const summaryView = priorSummary || failureView;
    renderProcessState(summaryView, failureView);
    views.setStatus(status.process, `Process request failed: ${error}`, true);
  } finally {
    renderShell();
    views.stopLoading(outputs.process);
    views.scrollIntoViewSmooth(outputs.process);
  }
}

function bindEvents() {
  document.getElementById("copy-env").addEventListener("click", copyEnvBlock);
  document.getElementById("run-diagnostics").addEventListener("click", handleDiagnostics);
  document.getElementById("runtime-check").addEventListener("click", handleRuntimeCheck);
  document.getElementById("bootstrap-models").addEventListener("click", handleBootstrapModels);
  document.getElementById("process-run").addEventListener("click", handleProcessRun);
  views.bindEnvInputs(fields, syncConfigurationProgress);
}

function boot() {
  state.hydrateDefaults();
  renderShell();
  syncConfigurationProgress();
  renderRuntimeState();
  renderProcessState();
  bindEvents();
  views.bindMobileStepToggle();
  views.bindPressEffects();
  views.bindRevealCards();
}

boot();
