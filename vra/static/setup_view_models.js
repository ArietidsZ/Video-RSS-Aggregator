function toTitleCase(value) {
  return String(value || "")
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function formatCount(value, noun) {
  if (value === null || value === undefined || value === "") {
    return `${noun} not reported`;
  }

  const count = Number(value);
  if (Number.isNaN(count)) {
    return `${value} ${noun}`;
  }

  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function describeScalar(value) {
  if (value === null || value === undefined || value === "") {
    return "Not reported";
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  return String(value);
}

function stringifyPayload(payload) {
  return JSON.stringify(payload, null, 2);
}

export function buildShellSummaryView(shellState) {
  const activeStep = shellState.steps.find((step) => step.id === shellState.activeStep);
  const activeStepLabel = activeStep ? toTitleCase(activeStep.id) : "Prerequisites";
  const blockedSteps = shellState.steps.filter((step) => step.status === "blocked");

  return {
    activeStepId: shellState.activeStep,
    activeStepLabel,
    readinessTitle: "Readiness",
    readinessBody: `Start with ${activeStepLabel.toLowerCase()}.`,
    blockerTitle: "Current blockers",
    blockerBody:
      blockedSteps.length > 0
        ? `${blockedSteps.length} step blockers need attention.`
        : "Diagnostics and runtime checks will surface anything still missing.",
    activeStepEyebrow: "Active step",
    activeStepTitle: `Focus: ${activeStepLabel}`,
    activeStepBody: "Move through the workbench one step at a time.",
  };
}

export function buildStaleSummaryLabel() {
  return "Showing the last successful result while the latest rerun failed.";
}

export function buildProcessSummaryView(result, stale = false) {
  if (!result || typeof result !== "object") {
    return {
      heading: stale ? "Previous successful run" : "Process result",
      title: "No successful process result yet",
      model: "Model not reported",
      frameCount: "Frames not reported",
      transcriptChars: "Transcript not reported",
      source: "Not reported",
      feedUrl: "Not reported",
      staleLabel: stale ? buildStaleSummaryLabel() : "",
      detailLines: [String(result)],
      raw: stringifyPayload(result),
    };
  }

  return {
    heading: stale ? "Previous successful run" : "Latest successful run",
    title: describeScalar(result.title || "Untitled source"),
    model: describeScalar(result.model),
    frameCount: formatCount(result.frame_count, "frame"),
    transcriptChars: formatCount(result.transcript_chars, "transcript char"),
    source: describeScalar(result.source_url),
    feedUrl: describeScalar(result.feed_url),
    staleLabel: stale ? buildStaleSummaryLabel() : "",
    detailLines: Object.entries(result)
      .filter(([, value]) => value !== null && value !== undefined)
      .map(([key, value]) => `${toTitleCase(key)}: ${
        typeof value === "object" ? JSON.stringify(value) : value
      }`),
    raw: stringifyPayload(result),
  };
}

export function buildVideoDetailSummaryView(detail) {
  const video = detail && typeof detail === "object" ? detail.video || detail : {};
  const summaries = Array.isArray(detail?.summaries) ? detail.summaries : [];
  const latest = summaries[0] || {};
  const transcript = detail?.transcript;
  const frames = Array.isArray(detail?.frames) ? detail.frames : [];

  return buildProcessSummaryView({
    title: video.title || video.source_url,
    model: latest.model,
    frame_count: latest.frame_count ?? frames.length,
    transcript_chars:
      latest.transcript_chars ??
      (typeof transcript?.text === "string" ? transcript.text.length : null),
    source_url: video.source_url,
    summary: latest.summary,
    key_points: latest.key_points || [],
    status: video.status,
    job_state: detail?.job_state,
  });
}

export function buildProcessFailureView(error, priorSummary = null) {
  const message = error instanceof Error ? error.message : String(error);

  return {
    heading: "Latest run failed",
    message,
    staleLabel: priorSummary ? buildStaleSummaryLabel() : "",
    detailLines: [`Error: ${message}`],
  };
}
