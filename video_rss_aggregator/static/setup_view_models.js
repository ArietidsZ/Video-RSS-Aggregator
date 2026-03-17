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

function collectProcessFrameCount(result) {
  if (typeof result.frame_count === "number") {
    return result.frame_count;
  }

  if (typeof result.frames_processed === "number") {
    return result.frames_processed;
  }

  if (typeof result.selected_frame_count === "number") {
    return result.selected_frame_count;
  }

  if (Array.isArray(result.frames)) {
    return result.frames.length;
  }

  return null;
}

function collectTranscriptChars(result) {
  if (typeof result.transcript_chars === "number") {
    return result.transcript_chars;
  }

  if (typeof result.transcript_length === "number") {
    return result.transcript_length;
  }

  if (typeof result.transcript === "string") {
    return result.transcript.length;
  }

  return null;
}

function collectProcessModel(result) {
  return (
    result.model ||
    result.model_name ||
    result.summary_model ||
    result.summary?.model_used ||
    result.selected_model ||
    "Model not reported"
  );
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

function buildDetailRows(payload) {
  return Object.entries(payload || {})
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => {
      if (typeof value === "object") {
        return `${toTitleCase(key)}: ${JSON.stringify(value, null, 2)}`;
      }

      return `${toTitleCase(key)}: ${value}`;
    });
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
    const summaryLines = [String(result)];

    return {
      heading: stale ? "Previous successful run" : "Process result",
      title: "No successful process result yet",
      model: "Model not reported",
      frameCount: "Frames not reported",
      transcriptChars: "Transcript not reported",
      source: "Not reported",
      feedUrl: "Not reported",
      staleLabel: stale ? buildStaleSummaryLabel() : "",
      detailLines: summaryLines,
      raw: stringifyPayload(result),
    };
  }

  const summaryLines = buildDetailRows(result);

  return {
    heading: stale ? "Previous successful run" : "Latest successful run",
    title: describeScalar(result.title || result.source_title || result.video_title || "Untitled source"),
    model: describeScalar(collectProcessModel(result)),
    frameCount: formatCount(collectProcessFrameCount(result), "frame"),
    transcriptChars: formatCount(collectTranscriptChars(result), "transcript char"),
    source: describeScalar(result.source_url || result.url || result.source_path),
    feedUrl: describeScalar(result.feed_url || result.rss_url),
    staleLabel: stale ? buildStaleSummaryLabel() : "",
    detailLines: summaryLines,
    raw: stringifyPayload(result),
  };
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
