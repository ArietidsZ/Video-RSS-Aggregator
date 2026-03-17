export function createSetupViews({ document, outputs, status, toastEl }) {
  let toastTimer = null;

  function appendTextNode(parent, tagName, text, className = "") {
    const node = document.createElement(tagName);
    if (className) {
      node.className = className;
    }
    node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  function buildSummaryGrid(rows) {
    const list = document.createElement("dl");
    list.className = "summary-grid";

    rows.forEach(([label, value]) => {
      const wrapper = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");

      term.textContent = label;
      detail.textContent = value;
      wrapper.append(term, detail);
      list.appendChild(wrapper);
    });

    return list;
  }

  function showToast(message, duration = 2200) {
    toastEl.textContent = message;
    toastEl.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastEl.classList.remove("visible");
    }, duration);
  }

  function setStatus(target, message, isError = false) {
    target.style.opacity = "0";
    requestAnimationFrame(() => {
      target.textContent = message;
      target.classList.toggle("error", isError);
      target.style.opacity = "";
      target.classList.remove("fade-in");
      void target.offsetWidth;
      target.classList.add("fade-in");
    });
  }

  function startLoading(preEl) {
    preEl.classList.add("loading");
  }

  function stopLoading(preEl) {
    preEl.classList.remove("loading");
  }

  function scrollIntoViewSmooth(el) {
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderEnvBlock(lines) {
    outputs.env.textContent = lines.join("\n");
  }

  function renderProgress(shellState) {
    document.querySelectorAll(".step-item").forEach((stepEl) => {
      const stepId = stepEl.dataset.stepId;
      const step = shellState.steps.find((item) => item.id === stepId);
      if (!step) {
        return;
      }

      const panel = document.getElementById(`step-panel-${stepId}`);
      const state = step.id === shellState.activeStep ? "active" : step.status;

      stepEl.setAttribute("data-state", state);
      if (panel) {
        panel.setAttribute("data-state", state);
      }
    });
  }

  function renderActiveStep(shellSummaryView) {
    const activeStepCopy = document.getElementById("active-step-copy");
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = shellSummaryView.activeStepEyebrow;
    const heading = document.createElement("h2");
    heading.textContent = shellSummaryView.activeStepTitle;
    const body = document.createElement("p");
    body.className = "muted";
    body.textContent = shellSummaryView.activeStepBody;
    activeStepCopy.replaceChildren(eyebrow, heading, body);
  }

  function renderShellState(shellState, shellSummaryView) {
    renderProgress(shellState);

    const readinessSummary = document.getElementById("readiness-summary");
    const blockerSummary = document.getElementById("blocker-summary");

    const readinessHeading = document.createElement("h2");
    readinessHeading.textContent = shellSummaryView.readinessTitle;
    const readinessBody = document.createElement("p");
    readinessBody.className = "muted";
    readinessBody.textContent = shellSummaryView.readinessBody;
    readinessSummary.replaceChildren(readinessHeading, readinessBody);

    const blockerHeading = document.createElement("h2");
    blockerHeading.textContent = shellSummaryView.blockerTitle;
    const blockerBody = document.createElement("p");
    blockerBody.className = "muted";
    blockerBody.textContent = shellSummaryView.blockerBody;
    blockerSummary.replaceChildren(blockerHeading, blockerBody);
    renderActiveStep(shellSummaryView);
  }

  function renderDiagnosticsSummary(report) {
    const blockers = Array.isArray(report?.blockers) ? report.blockers : [];
    const warningChecks = Array.isArray(report?.checks)
      ? report.checks.filter((check) => check.state !== "complete")
      : [];
    const lines = [
      report.ready
      || report?.state === "ready"
        ? "Prerequisites look ready for configuration and runtime checks."
        : "Diagnostics found blockers you should address before continuing.",
    ];

    if (blockers.length > 0) {
      lines.push(`Fixes: ${blockers.join(" | ")}`);
    } else if (Array.isArray(report?.missing) && report.missing.length > 0) {
      lines.push(`Missing: ${report.missing.join(", ")}`);
    }

    if (warningChecks.length > 0) {
      lines.push(`Checks: ${warningChecks.map((check) => `${check.label}: ${check.detail}`).join(" | ")}`);
    } else if (Array.isArray(report?.warnings) && report.warnings.length > 0) {
      lines.push(`Warnings: ${report.warnings.join(" | ")}`);
    }

    outputs.diagnostics.textContent = lines.join("\n");
  }

  function renderCommonFixes(view) {
    const container = document.getElementById("common-fixes");
    const heading = document.createElement("h2");
    heading.textContent = "Common fixes";

    const fixes = Array.isArray(view?.checks)
      ? view.checks.filter((check) => check.fix)
      : Array.isArray(view?.missing_models)
        ? view.missing_models.map((model) => ({
            label: model,
            detail: "Required locally before processing can start.",
            fix: `Pull or bootstrap ${model}`,
          }))
        : [];

    if (fixes.length === 0) {
      const body = document.createElement("p");
      body.className = "muted";
      body.textContent = view?.next_action || "Quick links and remediation steps will appear here.";
      container.replaceChildren(heading, body);
      return;
    }

    const list = document.createElement("ol");
    fixes.forEach((item) => {
      const row = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = item.label || "Fix";
      const detail = document.createElement("p");
      detail.className = "muted";
      detail.textContent = item.detail || item.fix;
      const fix = document.createElement("p");
      fix.textContent = item.fix;
      row.append(title, detail, fix);
      list.appendChild(row);
    });

    container.replaceChildren(heading, list);
  }

  function renderRuntimeSummary(runtimeView, runtime) {
    const summary = document.getElementById("runtime-summary");
    const isUnverified = !runtimeView || runtimeView.state === "unverified";

    if (isUnverified) {
      summary.setAttribute("data-state", "unverified");
      const eyebrow = appendTextNode(summary, "p", "Runtime summary", "eyebrow");
      const heading = appendTextNode(summary, "h3", "Runtime not checked yet");
      const body = appendTextNode(
        summary,
        "p",
        "Run a runtime check or bootstrap models to confirm connectivity and local model availability.",
        "muted",
      );
      summary.replaceChildren(eyebrow, heading, body);
      return;
    }

    const requiredModels = Array.isArray(runtime?.models) ? runtime.models : [];
    const localModels = Array.isArray(runtime?.local_models)
      ? runtime.local_models
      : Object.keys(runtime?.local_models || {});
    const databasePath = runtime?.database_path || "Database path not reported";
    const storageDir = runtime?.storage_dir || "Storage directory not reported";
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = "Runtime summary";
    const heading = document.createElement("h3");
    heading.textContent = runtimeView?.state === "ready" ? "Runtime ready for processing" : "Runtime needs local model setup";
    const body = document.createElement("p");
    body.className = "muted";
    body.textContent = runtimeView?.next_action || "Check runtime connectivity and model readiness.";
    const grid = buildSummaryGrid([
      ["Required models", `${requiredModels.length} configured`],
      ["Available locally", `${localModels.length} ready`],
      ["Database", databasePath],
      ["Storage", storageDir],
    ]);

    summary.setAttribute("data-state", runtimeView?.state === "ready" ? "complete" : "blocked");
    summary.replaceChildren(eyebrow, heading, body, grid);
  }

  function renderRuntimeDetails(runtime) {
    outputs.runtime.textContent = JSON.stringify(runtime, null, 2);
  }

  function renderProcessSummary(summaryView) {
    const summary = document.getElementById("process-summary");
    const isStale = Boolean(summaryView?.staleLabel);
    const isFailure = Boolean(summaryView?.message);

    summary.setAttribute(
      "data-state",
      isFailure ? "failed" : isStale ? "stale" : summaryView?.title ? "complete" : "unverified",
    );

    if (!summaryView) {
      const eyebrow = document.createElement("p");
      eyebrow.className = "eyebrow";
      eyebrow.textContent = "Processing summary";
      const heading = document.createElement("h3");
      heading.textContent = "No processing run yet";
      const body = document.createElement("p");
      body.className = "muted";
      body.textContent = "Run one source to see a human-readable result here.";
      summary.replaceChildren(eyebrow, heading, body);
      return;
    }

    const children = [];
    const eyebrow = document.createElement("p");
    eyebrow.className = "eyebrow";
    eyebrow.textContent = summaryView.heading || "Processing summary";
    children.push(eyebrow);

    const heading = document.createElement("h3");
    heading.textContent = summaryView.title || "Process run";
    children.push(heading);

    if (summaryView.staleLabel) {
      const staleChip = document.createElement("p");
      staleChip.className = "stale-chip";
      staleChip.textContent = summaryView.staleLabel;
      children.push(staleChip);
    }

    if (summaryView.message) {
      const message = document.createElement("p");
      message.className = "muted";
      message.textContent = summaryView.message;
      children.push(message);
    }

    children.push(
      buildSummaryGrid([
        ["Model", summaryView.model || "Model not reported"],
        ["Frames", summaryView.frameCount || "Frames not reported"],
        ["Transcript", summaryView.transcriptChars || "Transcript not reported"],
        ["Source", summaryView.source || "Not reported"],
        ["Feed", summaryView.feedUrl || "Not reported"],
      ]),
    );

    summary.replaceChildren(...children);
  }

  function renderProcessDetails(detailView) {
    if (typeof detailView === "string") {
      outputs.process.textContent = detailView;
      return;
    }

    if (Array.isArray(detailView?.detailLines)) {
      outputs.process.textContent = detailView.detailLines.join("\n");
      return;
    }

    if (detailView?.raw) {
      outputs.process.textContent = detailView.raw;
      return;
    }

    outputs.process.textContent = String(detailView || "");
  }

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

  function bindPressEffects() {
    document.querySelectorAll("button").forEach(addPressEffect);
  }

  function bindEnvInputs(fields, onUpdate) {
    Object.values(fields).forEach((node) => {
      if (!node || node === fields.processSource || node === fields.processTitle) {
        return;
      }
      node.addEventListener("input", onUpdate);
      node.addEventListener("change", onUpdate);
    });
  }

  function bindRevealCards() {
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
  }

  function bindMobileStepToggle() {
    const mobileStepToggle = document.getElementById("mobile-step-toggle");
    const progressRail = document.getElementById("setup-progress");

    if (!mobileStepToggle || !progressRail) {
      return;
    }

    function setExpanded(expanded) {
      progressRail.dataset.mobileOpen = expanded ? "true" : "false";
      mobileStepToggle.setAttribute("aria-expanded", String(expanded));
      mobileStepToggle.textContent = expanded ? "Hide setup steps" : "Open setup steps";
    }

    setExpanded(progressRail.dataset.mobileOpen === "true");
    mobileStepToggle.addEventListener("click", () => {
      setExpanded(progressRail.dataset.mobileOpen !== "true");
    });
  }

  return {
    bindEnvInputs,
    bindMobileStepToggle,
    bindPressEffects,
    bindRevealCards,
    renderActiveStep,
    renderDiagnosticsSummary,
    renderEnvBlock,
    renderCommonFixes,
    renderProcessDetails,
    renderProcessSummary,
    renderProgress,
    renderRuntimeDetails,
    renderRuntimeSummary,
    renderShellState,
    scrollIntoViewSmooth,
    setStatus,
    showToast,
    startLoading,
    stopLoading,
  };
}
