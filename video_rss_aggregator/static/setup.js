import { createSetupApi } from "./setup_api.js";
import { createSetupState } from "./setup_state.js";
import { buildShellSummaryView } from "./setup_view_models.js";
import { buildProcessSummaryView } from "./setup_view_models.js";
import { buildProcessFailureView } from "./setup_view_models.js";
import { createSetupViews } from "./setup_views.js";

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

function toRuntimeView(runtime) {
  return runtime.setup_view || { state: runtime.reachable ? "ready" : "blocked" };
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
  const requestDiagnostics = api.runDiagnostics;

  state.beginDiagnosticsCheck();
  renderShell();
  views.setStatus(status.diagnostics, "Running diagnostics...");
  views.startLoading(outputs.diagnostics);
  try {
    const report = await requestDiagnostics();
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

const handleDiagnosticsRun = handleDiagnostics;

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
    const runtime = await api.checkRuntime();
    const runtimeView = toRuntimeView(runtime);

    state.completeRuntimeCheck(runtimeView);
    renderRuntimeState(runtimeView, runtime);
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
      "Complete prerequisites and refresh the configuration before bootstrapping models.",
      true,
    );
    return;
  }

  renderShell();
  views.setStatus(status.runtime, "Bootstrapping models...");
  views.startLoading(outputs.runtime);
  try {
    const report = await api.bootstrapModels();
    const runtime = await api.checkRuntime();
    const runtimeView = toRuntimeView(runtime);

    state.completeRuntimeCheck(runtimeView);
    renderRuntimeState(runtimeView, { ...runtime, bootstrap: report });
    views.setStatus(status.runtime, "Models are ready.");
  } catch (error) {
    state.completeRuntimeCheck({ state: "blocked" });
    renderRuntimeState({ state: "blocked" }, { reachable: false, error: String(error) });
    views.setStatus(status.runtime, `Bootstrap failed: ${error}`, true);
  } finally {
    renderShell();
    views.stopLoading(outputs.runtime);
    views.scrollIntoViewSmooth(outputs.runtime);
  }
}

async function handleProcessRun() {
  const sourceUrl = fields.processSource.value.trim();
  if (!sourceUrl) {
    views.setStatus(status.process, "Please provide a source URL or local path.", true);
    return;
  }

  if (!state.beginProcess()) {
    renderShell();
    views.setStatus(status.process, "Complete the runtime check before processing a source.", true);
    return;
  }

  renderShell();
  views.setStatus(status.process, "Processing source... this may take a while.");
  views.startLoading(outputs.process);
  try {
    const result = await api.runProcess({
      source_url: sourceUrl,
      title: fields.processTitle.value.trim() || null,
    });
    const summaryView = buildProcessSummaryView(result);
    state.markProcessSuccess(result);
    renderProcessState(summaryView, summaryView);
    views.setStatus(status.process, "Processing completed.");
  } catch (error) {
    state.markProcessFailure(error);
    const priorSummary = state.steps.process.summary
      ? buildProcessSummaryView(state.steps.process.summary, true)
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
  document.getElementById("run-diagnostics").addEventListener("click", handleDiagnosticsRun);
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
