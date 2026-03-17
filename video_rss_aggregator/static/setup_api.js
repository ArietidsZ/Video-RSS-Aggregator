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
      throw new Error(typeof data === "string" ? data : JSON.stringify(data, null, 2));
    }

    return data;
  }

  function runDiagnostics() {
    return apiFetch("setup/diagnostics");
  }

  function checkRuntime() {
    return apiFetch("runtime");
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

  return {
    apiFetch,
    bootstrapModels,
    checkRuntime,
    runDiagnostics,
    runProcess,
  };
}
