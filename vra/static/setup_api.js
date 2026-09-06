const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;

export function createSetupApi({ fetchImpl, getAuthHeaders }) {
  async function apiFetch(path, options = {}) {
    const headers = {
      ...(options.headers || {}),
      ...getAuthHeaders(),
    };

    const response = await fetchImpl(path, {
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
      const detail = data && typeof data === "object" && "error" in data
        ? data.error
        : typeof data === "string"
          ? data
          : JSON.stringify(data, null, 2);
      throw new Error(detail);
    }

    return data;
  }

  function runDiagnostics() {
    return apiFetch("setup/diagnostics");
  }

  function checkRuntime() {
    return runDiagnostics();
  }

  function bootstrapModels() {
    return apiFetch("setup/bootstrap", { method: "POST" });
  }

  function runProcess(payload) {
    return apiFetch("process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  function getJob(jobId) {
    return apiFetch(`jobs/${encodeURIComponent(jobId)}`);
  }

  function getVideo(videoId) {
    return apiFetch(`videos/${encodeURIComponent(videoId)}`);
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function waitForJob(jobId, onTick = null) {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    let job = await getJob(jobId);
    while (!["done", "failed", "cancelled"].includes(job.state)) {
      if (Date.now() > deadline) {
        throw new Error(`job ${jobId} did not finish in time`);
      }
      await sleep(POLL_INTERVAL_MS);
      job = await getJob(jobId);
      if (onTick) {
        onTick(job);
      }
    }
    return job;
  }

  return {
    apiFetch,
    bootstrapModels,
    checkRuntime,
    getJob,
    getVideo,
    runDiagnostics,
    runProcess,
    waitForJob,
  };
}
