# Setup Studio UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current all-at-once setup page with a guided, beginner-first workbench that prevents setup mistakes while preserving the existing setup/runtime/process backend contract.

**Architecture:** Keep the FastAPI routes and setup endpoints intact, add narrow adapter-facing view-model helpers for diagnostics/runtime summaries, then rebuild the frontend around a guided workbench shell. Implement the browser logic as small vanilla JS modules loaded from one entrypoint so state transitions, API calls, and DOM rendering stay separate and testable.

**Tech Stack:** Python 3.11, FastAPI, packaged HTML templates, vanilla CSS, vanilla ES modules, pytest

---

Use `@superpowers:test-driven-development` for every implementation task and `@superpowers:verification-before-completion` before any completion claim or commit.

Spec reference: `docs/superpowers/specs/2026-03-17-setup-studio-ux-design.md`

## File Structure

- Create: `video_rss_aggregator/setup_view_models.py` - adapter-facing helpers that turn raw diagnostics/runtime payloads into step-ready summaries, blockers, and readiness state
- Modify: `video_rss_aggregator/api.py` - attach view-model fields to existing setup/runtime payloads without changing route ownership
- Modify: `video_rss_aggregator/templates/setup.html` - replace the current four-card grid with the guided workbench shell, progress rail, step panels, and details toggles
- Modify: `video_rss_aggregator/static/setup.css` - replace the current dark glass styling with the Workshop Manual theme and responsive workbench layout
- Create: `video_rss_aggregator/static/setup_api.js` - thin fetch wrappers for `setup/config`, `setup/diagnostics`, `runtime`, `setup/bootstrap`, and `process`
- Create: `video_rss_aggregator/static/setup_state.js` - client-side step order, state transitions, gating rules, and stale-result tracking
- Create: `video_rss_aggregator/static/setup_view_models.js` - client-side formatting helpers for env preview, process summaries, stale labels, and disclosure-friendly text
- Create: `video_rss_aggregator/static/setup_views.js` - DOM rendering helpers for the progress rail, active step panel, summaries, blockers, and details sections
- Modify: `video_rss_aggregator/static/setup.js` - convert the current monolith into the boot module that hydrates defaults and wires the new modules together
- Create: `tests/test_setup_view_models.py` - focused Python tests for diagnostics/runtime step-view helpers
- Modify: `tests/test_api_setup.py` - setup route payload coverage and root-path behavior
- Create: `tests/test_setup_studio_contract.py` - setup page shell, CSS, and JS asset contract coverage
- Modify: `tests/adapters/test_api_app.py` - HTTP shape coverage for additive setup/runtime view-model fields

## Chunk 1: Adapter View Models

### Task 1: Add diagnostics and runtime view-model helpers

**Files:**
- Create: `video_rss_aggregator/setup_view_models.py`
- Test: `tests/test_setup_view_models.py`

- [ ] **Step 1: Write the failing helper tests**

```python
from video_rss_aggregator.setup_view_models import (
    build_diagnostics_view,
    build_runtime_view,
)


def test_build_diagnostics_view_marks_missing_dependencies_blocked() -> None:
    report = {
        "ready": False,
        "platform": {
            "python_version": "3.11.9",
            "python_executable": "/usr/bin/python3",
        },
        "dependencies": {
            "ffmpeg": {"available": False, "command": None},
            "ffprobe": {"available": True, "command": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "/usr/bin/yt-dlp"},
            "ollama": {"reachable": False, "error": "connection refused"},
        },
    }

    view = build_diagnostics_view(report)

    assert view["state"] == "blocked"
    assert view["blockers"] == ["Install FFmpeg and add it to PATH", "Start Ollama so the local API is reachable"]
    assert view["checks"][0] == {
        "id": "python",
        "label": "Python 3.11+",
        "state": "complete",
        "detail": "3.11.9 via /usr/bin/python3",
        "fix": None,
    }
    assert view["checks"][1]["id"] == "ffmpeg"
    assert view["checks"][1]["state"] == "blocked"


def test_build_diagnostics_view_marks_ready_path_complete() -> None:
    report = {
        "ready": True,
        "platform": {
            "python_version": "3.11.9",
            "python_executable": "/usr/bin/python3",
        },
        "dependencies": {
            "ffmpeg": {"available": True, "command": "/usr/bin/ffmpeg"},
            "ffprobe": {"available": True, "command": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "/usr/bin/yt-dlp"},
            "ollama": {"reachable": True, "error": None},
        },
    }

    view = build_diagnostics_view(report)

    assert view["state"] == "ready"
    assert view["blockers"] == []
    assert view["next_action"] == "Continue to configuration"


def test_build_runtime_view_marks_missing_models_blocked() -> None:
    runtime = {
        "reachable": True,
        "local_models": {"qwen3.5:2b-q4_K_M": {}},
        "models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
    }

    view = build_runtime_view(runtime)

    assert view["state"] == "blocked"
    assert view["missing_models"] == ["qwen3.5:4b-q4_K_M"]
    assert view["next_action"] == "Bootstrap required models"


def test_build_runtime_view_marks_ready_path_complete() -> None:
    runtime = {
        "reachable": True,
        "local_models": {
            "qwen3.5:4b-q4_K_M": {},
            "qwen3.5:2b-q4_K_M": {},
        },
        "models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
    }

    view = build_runtime_view(runtime)

    assert view["state"] == "ready"
    assert view["missing_models"] == []
    assert view["next_action"] == "Run the first processing test"
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `pytest tests/test_setup_view_models.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing helper names.

- [ ] **Step 3: Create the helper module with explicit step summaries**

```python
def _python_check(platform_data: dict[str, object]) -> dict[str, object]:
    return {
        "id": "python",
        "label": "Python 3.11+",
        "state": "complete",
        "detail": f'{platform_data["python_version"]} via {platform_data["python_executable"]}',
        "fix": None,
    }


def _tool_check(tool_id: str, label: str, payload: dict[str, object], fix: str) -> dict[str, object]:
    available = bool(payload.get("available"))
    return {
        "id": tool_id,
        "label": label,
        "state": "complete" if available else "blocked",
        "detail": payload.get("command") or "Not found on PATH",
        "fix": None if available else fix,
    }


def _ollama_check(payload: dict[str, object]) -> dict[str, object]:
    reachable = bool(payload.get("reachable"))
    return {
        "id": "ollama",
        "label": "Ollama API",
        "state": "complete" if reachable else "blocked",
        "detail": payload.get("version") or payload.get("error") or "Not reachable",
        "fix": None if reachable else "Start Ollama so the local API is reachable",
    }


def build_diagnostics_view(report: dict[str, object]) -> dict[str, object]:
    checks = [
        _python_check(report["platform"]),
        _tool_check("ffmpeg", "FFmpeg", report["dependencies"]["ffmpeg"], "Install FFmpeg and add it to PATH"),
        _tool_check("ffprobe", "FFprobe", report["dependencies"]["ffprobe"], "Install FFprobe and add it to PATH"),
        _tool_check("yt_dlp", "yt-dlp", report["dependencies"]["yt_dlp"], "Install yt-dlp so downloads can run"),
        _ollama_check(report["dependencies"]["ollama"]),
    ]
    blockers = [check["fix"] for check in checks if check["state"] == "blocked"]
    return {
        "state": "ready" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "next_action": "Continue to configuration" if not blockers else "Resolve the blocked checks and run diagnostics again",
    }


def build_runtime_view(runtime: dict[str, object]) -> dict[str, object]:
    available = set(runtime.get("local_models", {}).keys())
    required = list(runtime.get("models", []))
    missing_models = [model for model in required if model not in available]
    return {
        "state": "ready" if runtime.get("reachable") and not missing_models else "blocked",
        "missing_models": missing_models,
        "next_action": "Check runtime" if not runtime.get("reachable") else (
            "Bootstrap required models" if missing_models else "Run the first processing test"
        ),
    }
```

- [ ] **Step 4: Re-run the focused helper tests**

Run: `pytest tests/test_setup_view_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the helper module**

```bash
git add tests/test_setup_view_models.py video_rss_aggregator/setup_view_models.py
git commit -m "feat: add setup view model helpers"
```

### Task 2: Thread view-model helpers through existing setup/runtime routes

**Files:**
- Modify: `video_rss_aggregator/api.py`
- Modify: `tests/test_api_setup.py`
- Modify: `tests/adapters/test_api_app.py`
- Test: `tests/test_setup_view_models.py`

- [ ] **Step 1: Add failing API-shape assertions**

```python
def test_gui_and_setup_routes() -> None:
    diagnostics = client.get("/setup/diagnostics")
    diag_payload = diagnostics.json()
    assert diag_payload["setup_view"]["state"] in {"ready", "blocked"}
    assert [check["id"] for check in diag_payload["setup_view"]["checks"]] == [
        "python",
        "ffmpeg",
        "ffprobe",
        "yt_dlp",
        "ollama",
    ]
    assert "platform" in diag_payload
    assert "dependencies" in diag_payload


def test_routes_delegate_to_runtime_use_cases_and_keep_http_shapes() -> None:
    runtime_response = client.get("/runtime")
    assert runtime_response.json()["setup_view"] == {
        "state": "blocked",
        "missing_models": ["qwen:min"],
        "next_action": "Bootstrap required models",
    }
```

- [ ] **Step 2: Run the affected adapter tests and verify they fail**

Run: `pytest tests/test_api_setup.py tests/adapters/test_api_app.py -q`
Expected: FAIL with missing `setup_view` keys.

- [ ] **Step 3: Attach additive view-model fields in `video_rss_aggregator/api.py`**

```python
from video_rss_aggregator.setup_view_models import (
    build_diagnostics_view,
    build_runtime_view,
)


diagnostics_payload = {
    "platform": {...},
    "dependencies": {...},
    "ready": ffmpeg_ok and ffprobe_ok and ytdlp_ok and ollama_ok,
}
diagnostics_payload["setup_view"] = build_diagnostics_view(diagnostics_payload)
return diagnostics_payload


runtime_payload = await _runtime(request).use_cases.get_runtime_status.execute()
return {
    **runtime_payload,
    "setup_view": build_runtime_view(runtime_payload),
}
```

Keep `POST /setup/bootstrap` compatible with the current raw response shape. The frontend should treat bootstrap as an action trigger and then perform a follow-up `GET /runtime` refresh to obtain the updated `setup_view` payload.

- [ ] **Step 4: Re-run the focused adapter and helper tests**

Run: `pytest tests/test_setup_view_models.py tests/test_api_setup.py tests/adapters/test_api_app.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the API contract update**

```bash
git add video_rss_aggregator/api.py tests/test_api_setup.py tests/adapters/test_api_app.py tests/test_setup_view_models.py video_rss_aggregator/setup_view_models.py
git commit -m "feat: add setup route view models"
```

## Chunk 2: Guided Workbench Shell

### Task 3: Replace the four-card page with the guided workbench template

**Files:**
- Modify: `video_rss_aggregator/templates/setup.html`
- Create: `tests/test_setup_studio_contract.py`

- [ ] **Step 1: Add a failing shell-contract test**

```python
client = TestClient(create_app(_build_runtime(Config())))


def test_setup_home_renders_guided_workbench_shell() -> None:
    home = client.get("/")

    assert 'id="setup-workbench"' in home.text
    assert 'id="setup-progress"' in home.text
    assert 'id="readiness-summary"' in home.text
    assert 'id="blocker-summary"' in home.text
    assert 'id="common-fixes"' in home.text
    assert 'data-step-id="prerequisites"' in home.text
    assert 'data-step-id="configuration"' in home.text
    assert 'id="step-panel-process"' in home.text
    assert 'id="advanced-config"' in home.text
    assert 'id="mobile-step-toggle"' in home.text
```

- [ ] **Step 2: Run the route test and verify it fails**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_home_renders_guided_workbench_shell -q`
Expected: FAIL because the current template still renders the four-card grid.

- [ ] **Step 3: Rewrite `setup.html` around a progress rail and active step panels**

```html
<main class="shell" id="setup-workbench">
  <section class="hero">...</section>
  <section class="workbench">
    <aside class="progress-rail" id="setup-progress">
      <div id="readiness-summary"></div>
      <div id="blocker-summary"></div>
      <ol class="step-list">
        <li class="step-item" data-step-id="prerequisites"></li>
        <li class="step-item" data-step-id="configuration"></li>
        <li class="step-item" data-step-id="runtime"></li>
        <li class="step-item" data-step-id="process"></li>
      </ol>
      <nav id="common-fixes"></nav>
    </aside>
    <section class="step-stage">
      <button id="mobile-step-toggle" type="button"></button>
      <header id="active-step-copy"></header>
      <article class="step-panel" id="step-panel-prerequisites"></article>
      <article class="step-panel" id="step-panel-configuration"></article>
      <article class="step-panel" id="step-panel-runtime"></article>
      <article class="step-panel" id="step-panel-process"></article>
    </section>
  </section>
</main>
```

- [ ] **Step 4: Re-run the route test**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_home_renders_guided_workbench_shell -q`
Expected: PASS.

- [ ] **Step 5: Commit the shell rewrite**

```bash
git add video_rss_aggregator/templates/setup.html tests/test_setup_studio_contract.py
git commit -m "feat: add guided setup workbench shell"
```

### Task 4: Apply the Workshop Manual visual system and responsive layout

**Files:**
- Modify: `video_rss_aggregator/static/setup.css`
- Modify: `tests/test_setup_studio_contract.py`

- [ ] **Step 1: Add a failing static-asset contract test for the new theme hooks**

```python
client = TestClient(create_app(_build_runtime(Config())))


def test_setup_css_uses_workshop_manual_tokens() -> None:
    css = client.get("/static/setup.css")

    assert "--paper-base" in css.text
    assert ".workbench" in css.text
    assert ".progress-rail" in css.text
    assert ".step-item[data-state=\"blocked\"]" in css.text
    assert ".mobile-step-toggle" in css.text
    assert "@media (max-width: 860px)" in css.text
    assert "#6366f1" not in css.text
```

- [ ] **Step 2: Run the CSS contract test and verify it fails**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_css_uses_workshop_manual_tokens -q`
Expected: FAIL because the current stylesheet still contains the dark purple theme.

- [ ] **Step 3: Replace the token set and add workbench/layout/state styles**

```css
:root {
  --paper-base: #f5f1e8;
  --paper-panel: #fffaf2;
  --ink-strong: #2f2a24;
  --ink-soft: #665d52;
  --accent-olive: #6b7a40;
  --accent-amber: #b86a2b;
  --danger-rust: #b44c37;
}

.workbench {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr);
  gap: 24px;
}

.mobile-step-toggle {
  display: none;
}

.progress-rail,
.step-panel,
.detail-drawer {
  background: var(--paper-panel);
  border: 1px solid rgba(47, 42, 36, 0.12);
}

@media (max-width: 860px) {
  .workbench {
    grid-template-columns: 1fr;
  }

  .mobile-step-toggle {
    display: inline-flex;
  }
}
```

- [ ] **Step 4: Re-run the CSS contract test**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_css_uses_workshop_manual_tokens -q`
Expected: PASS.

- [ ] **Step 5: Commit the theme and layout CSS**

```bash
git add video_rss_aggregator/static/setup.css tests/test_setup_studio_contract.py
git commit -m "feat: restyle setup studio as guided workbench"
```

## Chunk 3: Browser State and Rendering

### Task 5: Split the setup script into entrypoint, API, state, and view modules

**Files:**
- Create: `video_rss_aggregator/static/setup_api.js`
- Create: `video_rss_aggregator/static/setup_state.js`
- Create: `video_rss_aggregator/static/setup_view_models.js`
- Create: `video_rss_aggregator/static/setup_views.js`
- Modify: `video_rss_aggregator/static/setup.js`
- Modify: `video_rss_aggregator/templates/setup.html`
- Modify: `tests/test_setup_studio_contract.py`

- [ ] **Step 1: Add a failing module-loading contract test**

```python
def test_setup_assets_load_as_modules_without_auto_running_diagnostics() -> None:
    home = client.get("/")
    js = client.get("/static/setup.js")
    api_js = client.get("/static/setup_api.js")
    state_js = client.get("/static/setup_state.js")
    view_models_js = client.get("/static/setup_view_models.js")
    views_js = client.get("/static/setup_views.js")

    assert 'type="module"' in home.text
    assert 'import { createSetupApi } from "./setup_api.js";' in js.text
    assert 'import { createSetupState } from "./setup_state.js";' in js.text
    assert 'import { buildProcessSummaryView } from "./setup_view_models.js";' in js.text
    assert 'import { createSetupViews } from "./setup_views.js";' in js.text
    assert 'runDiagnostics();' not in js.text
    assert api_js.status_code == 200
    assert 'export function createSetupState' in state_js.text
    assert 'export function buildProcessSummaryView' in view_models_js.text
    assert 'export function createSetupViews' in views_js.text
```

- [ ] **Step 2: Run the asset contract test and verify it fails**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_assets_load_as_modules_without_auto_running_diagnostics -q`
Expected: FAIL because only the monolithic `setup.js` exists and it still auto-runs diagnostics.

- [ ] **Step 3: Create the new static modules and convert the entrypoint to `type="module"`**

```javascript
// setup_api.js
async function apiFetch(path, authHeaders, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...authHeaders(),
    },
  });
  const bodyText = await response.text();
  const data = bodyText ? JSON.parse(bodyText) : null;
  if (!response.ok) {
    throw new Error(typeof data === "string" ? data : JSON.stringify(data, null, 2));
  }
  return data;
}

export function createSetupApi(authHeaders) {
  return {
    loadConfig: () => apiFetch("setup/config", authHeaders),
    runDiagnostics: () => apiFetch("setup/diagnostics", authHeaders),
    loadRuntime: () => apiFetch("runtime", authHeaders),
    bootstrapModels: () => apiFetch("setup/bootstrap", authHeaders, { method: "POST" }),
    processSource: (payload) => apiFetch("process", authHeaders, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  };
}

// setup_state.js
export function createSetupState() {
  return { activeStep: "prerequisites", steps: {} };
}

// setup_view_models.js
export function buildProcessSummaryView(result) {
  return {
    title: result.title,
    modelUsed: result.summary.model_used,
    frameCount: result.frame_count,
  };
}

// setup_views.js
export function createSetupViews(root) {
  return {
    renderProgress() {},
    renderActiveStep() {},
    renderDetails() {},
  };
}

// setup.js
import { createSetupApi } from "./setup_api.js";
import { createSetupState } from "./setup_state.js";
import { buildProcessSummaryView } from "./setup_view_models.js";
import { createSetupViews } from "./setup_views.js";
```

- [ ] **Step 4: Re-run the module-loading contract test**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_assets_load_as_modules_without_auto_running_diagnostics -q`
Expected: PASS.

- [ ] **Step 5: Commit the modular frontend structure**

```bash
git add video_rss_aggregator/templates/setup.html video_rss_aggregator/static/setup.js video_rss_aggregator/static/setup_api.js video_rss_aggregator/static/setup_state.js video_rss_aggregator/static/setup_view_models.js video_rss_aggregator/static/setup_views.js tests/test_setup_studio_contract.py
git commit -m "refactor: split setup studio frontend modules"
```

### Task 6: Implement the step-state controller and gating rules

**Files:**
- Modify: `video_rss_aggregator/static/setup_state.js`
- Modify: `tests/test_setup_studio_contract.py`

- [ ] **Step 1: Add a failing state-controller contract test**

```python
def test_setup_state_module_exports_guardrail_transitions() -> None:
    js = client.get("/static/setup_state.js")

    assert 'createSetupState' in js.text
    assert 'unverified' in js.text
    assert 'blocked' in js.text
    assert 'running' in js.text
    assert 'failed' in js.text
    assert 'complete' in js.text
    assert 'beginDiagnosticsCheck' in js.text
    assert 'applyDiagnosticsResult' in js.text
    assert 'completeRuntimeCheck' in js.text
    assert 'markProcessSuccess' in js.text
    assert 'markProcessFailure' in js.text
    assert 'getRecommendedNextStep' in js.text
```

- [ ] **Step 2: Run the state-controller test and verify it fails**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_state_module_exports_guardrail_transitions -q`
Expected: FAIL because the state module does not implement the full guardrail contract yet.

- [ ] **Step 3: Implement the state machine with explicit dependency gating**

```javascript
const STEP_ORDER = ["prerequisites", "configuration", "runtime", "process"];

export function createSetupState() {
  const steps = {
    prerequisites: { state: "unverified" },
    configuration: { state: "ready" },
    runtime: { state: "blocked" },
    process: { state: "blocked", stale: false, summary: null },
  };

  return {
    steps,
    beginDiagnosticsCheck() {
      steps.prerequisites.state = "running";
    },
    applyDiagnosticsResult(view) {
      steps.prerequisites.state = view.state === "ready" ? "complete" : "blocked";
      steps.configuration.state = view.state === "ready" ? "ready" : "blocked";
      steps.runtime.state = view.state === "ready" ? "unverified" : "blocked";
      steps.process.state = "blocked";
    },
    markConfigurationComplete() {
      steps.configuration.state = "complete";
      if (steps.runtime.state === "blocked") {
        steps.runtime.state = "unverified";
      }
    },
    beginRuntimeCheck() {
      steps.runtime.state = "running";
    },
    completeRuntimeCheck(view) {
      steps.runtime.state = view.state === "ready" ? "complete" : "blocked";
      steps.process.state = view.state === "ready" ? "ready" : "blocked";
    },
    beginProcess() {
      steps.process.state = "running";
    },
    markProcessSuccess(summary) {
      steps.process = { state: "complete", stale: false, summary };
    },
    markProcessFailure(error) {
      steps.process = { ...steps.process, state: "failed", stale: Boolean(steps.process.summary), error };
    },
    getRecommendedNextStep() {
      return STEP_ORDER.find((stepId) => ["ready", "unverified", "blocked"].includes(steps[stepId].state));
    },
  };
}
```

- [ ] **Step 4: Re-run the state-controller test**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_state_module_exports_guardrail_transitions -q`
Expected: PASS.

- [ ] **Step 5: Commit the state controller**

```bash
git add video_rss_aggregator/static/setup_state.js tests/test_setup_studio_contract.py
git commit -m "feat: add setup step state controller"
```

### Task 7: Render human-readable summaries, details, and stale-result UI

**Files:**
- Modify: `video_rss_aggregator/static/setup_view_models.js`
- Modify: `video_rss_aggregator/static/setup_views.js`
- Modify: `video_rss_aggregator/static/setup.js`
- Modify: `video_rss_aggregator/templates/setup.html`
- Modify: `video_rss_aggregator/static/setup.css`
- Modify: `tests/test_setup_studio_contract.py`

- [ ] **Step 1: Add failing shell and asset tests for summary/detail hooks**

```python
def test_setup_home_exposes_summary_and_detail_hooks() -> None:
    home = client.get("/")
    entry_js = client.get("/static/setup.js")
    view_models_js = client.get("/static/setup_view_models.js")
    views_js = client.get("/static/setup_views.js")

    assert 'data-state="unverified"' in home.text
    assert 'id="runtime-summary"' in home.text
    assert 'id="process-summary"' in home.text
    assert 'id="advanced-config"' in home.text
    assert 'id="runtime-details"' in home.text
    assert 'id="process-details"' in home.text
    assert 'export function buildProcessSummaryView' in view_models_js.text
    assert 'export function buildProcessFailureView' in view_models_js.text
    assert 'export function buildStaleSummaryLabel' in view_models_js.text
    assert 'renderProcessSummary' in views_js.text
    assert 'renderDiagnosticsSummary' in views_js.text
    assert 'renderProgress' in views_js.text
    assert 'renderActiveStep' in views_js.text
    assert 'renderRuntimeSummary' in views_js.text
    assert 'renderRuntimeDetails' in views_js.text
    assert 'renderProcessDetails' in views_js.text
    assert 'handleDiagnosticsRun' in entry_js.text
    assert 'handleRuntimeCheck' in entry_js.text
    assert 'handleProcessRun' in entry_js.text
```

- [ ] **Step 2: Run the summary/detail test and verify it fails**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_home_exposes_summary_and_detail_hooks -q`
Expected: FAIL because the summary/detail containers and helper exports do not exist yet.

- [ ] **Step 3: Implement client summary builders and DOM renderers**

```javascript
export function buildProcessSummaryView(result, stale = false) {
  return {
    title: result.title,
    modelUsed: result.summary.model_used,
    frameCount: result.frame_count,
    transcriptChars: result.transcript_chars,
    staleLabel: stale ? "Showing last successful result" : null,
  };
}

export function buildStaleSummaryLabel() {
  return "Showing last successful result";
}

export function buildProcessFailureView(error, priorSummary = null) {
  return {
    title: priorSummary ? priorSummary.title : "No successful run yet",
    staleLabel: priorSummary ? buildStaleSummaryLabel() : null,
    message: priorSummary ? String(error) : `Processing failed: ${String(error)}`,
  };
}

export function createSetupViews(root) {
  return {
    renderProgress(steps) {
      Object.entries(steps).forEach(([stepId, step]) => {
        root.querySelector(`[data-step-id="${stepId}"]`)?.setAttribute("data-state", step.state);
      });
    },
    renderActiveStep(stepId) {
      root.querySelector("#active-step-copy").setAttribute("data-active-step", stepId);
    },
    renderDiagnosticsSummary(view) {
      root.querySelector("#blocker-summary").textContent = view.blockers.join(" | ") || view.next_action;
    },
    renderRuntimeSummary(view) {
      root.querySelector("#runtime-summary").textContent = view.next_action;
    },
    renderRuntimeDetails(runtime) {
      root.querySelector("#runtime-details").textContent = JSON.stringify({
        models: runtime.models,
        local_models: runtime.local_models,
        reachable: runtime.reachable,
      }, null, 2);
    },
    renderProcessSummary(summary) {
      root.querySelector("#process-summary").textContent = `${summary.title} · ${summary.modelUsed} · ${summary.frameCount} frames`;
      root.querySelector("#process-stale-label").textContent = summary.staleLabel || "";
    },
    renderProcessDetails(resultOrError) {
      root.querySelector("#process-details").textContent = JSON.stringify(resultOrError, null, 2);
    },
  };
}
```

- [ ] **Step 4: Wire the entrypoint and template hooks together**

```javascript
async function handleDiagnosticsRun() {
  state.beginDiagnosticsCheck();
  views.renderProgress(state.steps);
  views.renderActiveStep("prerequisites");
  const diagnostics = await api.runDiagnostics();
  state.applyDiagnosticsResult(diagnostics.setup_view);
  views.renderProgress(state.steps);
  views.renderDiagnosticsSummary(diagnostics.setup_view);
}

async function handleRuntimeCheck() {
  state.beginRuntimeCheck();
  views.renderProgress(state.steps);
  views.renderActiveStep("runtime");
  const runtime = await api.loadRuntime();
  state.completeRuntimeCheck(runtime.setup_view);
  views.renderProgress(state.steps);
  views.renderRuntimeSummary(runtime.setup_view);
  views.renderRuntimeDetails(runtime);
}

async function handleBootstrap() {
  await api.bootstrapModels();
  const runtime = await api.loadRuntime();
  state.completeRuntimeCheck(runtime.setup_view);
  views.renderRuntimeSummary(runtime.setup_view);
  views.renderRuntimeDetails(runtime);
}

async function handleProcessRun() {
  state.beginProcess();
  views.renderProgress(state.steps);
  views.renderActiveStep("process");
  try {
    const result = await api.processSource(readProcessForm());
    state.markProcessSuccess(result);
    views.renderProgress(state.steps);
    views.renderProcessSummary(buildProcessSummaryView(result));
    views.renderProcessDetails(result);
  } catch (error) {
    state.markProcessFailure(error);
    views.renderProgress(state.steps);
    const failureView = buildProcessFailureView(error, state.steps.process.summary);
    views.renderProcessSummary(
      state.steps.process.summary
        ? buildProcessSummaryView(state.steps.process.summary, true)
        : {
            title: failureView.title,
            modelUsed: "unavailable",
            frameCount: 0,
            transcriptChars: 0,
            staleLabel: failureView.staleLabel,
          }
    );
    views.renderProcessDetails({ error: failureView.message });
  }
}
```

- [ ] **Step 5: Re-run the summary/detail test**

Run: `pytest tests/test_setup_studio_contract.py::test_setup_home_exposes_summary_and_detail_hooks -q`
Expected: PASS.

- [ ] **Step 6: Perform a concrete manual browser smoke for the guided flow**

Run: `python -m vra serve --bind 127.0.0.1:8080`
Expected:
- the page opens with prerequisites in `unverified` state and diagnostics idle
- clicking `Run checks` moves prerequisites through `running` to `blocked` or `complete`
- blocked prerequisites keep runtime and process visibly blocked
- after runtime succeeds, the process step becomes ready
- a successful sample run shows title, model, frame count, and transcript size in `#process-summary`
- rerunning with a bad URL keeps the previous successful summary visible with the stale label text

- [ ] **Step 7: Commit the summary/detail rendering flow**

```bash
git add video_rss_aggregator/static/setup_view_models.js video_rss_aggregator/static/setup_views.js video_rss_aggregator/static/setup.js video_rss_aggregator/templates/setup.html video_rss_aggregator/static/setup.css tests/test_setup_studio_contract.py
git commit -m "feat: add guided setup summaries and detail views"
```

## Chunk 4: Regression Net and Final Verification

### Task 8: Lock in additive route coverage and guided-UX regressions

**Files:**
- Modify: `tests/test_api_setup.py`
- Modify: `tests/test_setup_studio_contract.py`
- Modify: `tests/adapters/test_api_app.py`

- [ ] **Step 1: Add failing regression assertions for preserved contracts and guided behavior**

```python
def test_runtime_requires_api_key_when_enabled() -> None:
    authorized = client.get("/runtime", headers={"X-API-Key": "secret"})
    assert authorized.json()["setup_view"]["state"] == "blocked"


def test_setup_config_omits_api_key_and_uses_current_bind_in_commands() -> None:
    payload = client.get("/setup/config").json()
    assert payload["quick_commands"]["serve"].startswith("python -m vra serve --bind")
    assert "api_key" not in payload


def test_setup_home_keeps_advanced_sections_collapsed_by_default() -> None:
    home = client.get("/")
    assert '<details id="advanced-config">' in home.text
    assert '<details id="runtime-details">' in home.text
    assert '<details id="process-details">' in home.text
```

- [ ] **Step 2: Run the focused regression suite and verify failures**

Run: `pytest tests/test_setup_view_models.py tests/test_api_setup.py tests/test_setup_studio_contract.py tests/adapters/test_api_app.py -q`
Expected: FAIL until the route payloads, shell hooks, and additive response shapes all line up.

- [ ] **Step 3: Update the concrete fixtures and assertions, not generic placeholders**

```python
runtime_status = {
    "ollama_version": "0.6.0",
    "local_models": {"qwen": {"size": 1}},
    "reachable": True,
    "database_path": ".data/runtime.db",
    "storage_dir": ".data",
    "models": ["qwen", "qwen:min"],
}

assert runtime_response.json()["setup_view"] == {
    "state": "blocked",
    "missing_models": ["qwen:min"],
    "next_action": "Bootstrap required models",
}
```

Update `tests/adapters/test_api_app.py` for the additive `setup_view` runtime field, keep `tests/test_api_setup.py` focused on route payload compatibility, and keep `tests/test_setup_studio_contract.py` focused on shell/assets/details-disclosure behavior.

- [ ] **Step 4: Re-run the focused regression suite**

Run: `pytest tests/test_setup_view_models.py tests/test_api_setup.py tests/test_setup_studio_contract.py tests/adapters/test_api_app.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the regression net**

```bash
git add tests/test_api_setup.py tests/test_setup_studio_contract.py tests/adapters/test_api_app.py
git commit -m "test: cover guided setup studio regressions"
```

### Task 9: Run full verification and document the execution handoff

**Files:**
- None

- [ ] **Step 1: Run the full automated test suite**

Run: `pytest -q`
Expected: PASS with `0 failed`.

- [ ] **Step 2: Run the linter**

Run: `ruff check .`
Expected: PASS with no lint errors.

- [ ] **Step 3: Repeat the manual smoke with explicit checkpoints**

Run: `python -m vra serve --bind 127.0.0.1:8080`
Expected:
- load `/` and confirm diagnostics do not start automatically
- run diagnostics and verify blocked prerequisites name the missing dependency or complete successfully
- check runtime, then bootstrap if models are missing, and verify the runtime summary changes afterward
- run the sample URL and confirm the process summary shows title/model/frame count/transcript chars
- rerun with an invalid URL or path and confirm the stale summary label remains visible on the last success

- [ ] **Step 4: Capture any manual follow-up notes directly in the PR or execution notes, not in new product code**

```text
Manual smoke checklist:
- diagnostics idle on first load
- blocked prerequisite message names the missing dependency
- runtime step shows missing-model guidance before bootstrap
- process failure preserves stale summary label on retry failure
```

- [ ] **Step 5: Verify the branch is clean after the prior task-level commits**

Run: `git status --short`
Expected: no output.
