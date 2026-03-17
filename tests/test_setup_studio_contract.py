from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

from fastapi.testclient import TestClient

from adapter_api import create_app
from core_config import Config
from video_rss_aggregator.bootstrap import AppRuntime, AppUseCases


@dataclass
class _AsyncValue:
    value: Any

    async def execute(self, *_args, **_kwargs) -> Any:
        return self.value


def _build_runtime(config: Config) -> AppRuntime:
    runtime_status = {
        "ollama_version": "0.6.0",
        "local_models": {"qwen3.5:2b-q4_K_M": {}},
        "reachable": True,
        "database_path": config.database_path,
        "storage_dir": config.storage_dir,
        "models": list(config.model_priority),
    }
    return AppRuntime(
        config=config,
        use_cases=AppUseCases(
            get_runtime_status=_AsyncValue(runtime_status),
            bootstrap_runtime=_AsyncValue({"models": ["qwen3.5:2b-q4_K_M"]}),
            ingest_feed=cast(Any, _AsyncValue(None)),
            process_source=cast(Any, _AsyncValue(None)),
            render_rss_feed=_AsyncValue("<rss></rss>"),
        ),
        close=lambda: None,
    )


client = TestClient(create_app(_build_runtime(Config())))


def _is_javascript_content_type(content_type: str) -> bool:
    return content_type.startswith(("text/javascript", "application/javascript"))


def class_tokens_for_id(html: str, element_id: str) -> set[str]:
    tag_match = re.search(rf'<[^>]*\bid="{re.escape(element_id)}"[^>]*>', html)
    assert tag_match is not None

    class_match = re.search(r'\bclass="([^"]+)"', tag_match.group(0))
    assert class_match is not None

    return set(class_match.group(1).split())


def test_setup_css_supports_shell_and_disclosure_layout() -> None:
    css = client.get("/static/setup.css")

    assert css.status_code == 200
    assert "--paper-base" in css.text
    assert ".shell" in css.text
    assert ".progress-rail" in css.text
    assert ".detail-drawer" in css.text
    assert ".summary-card" in css.text
    assert "@media (max-width: 860px)" in css.text
    assert '#setup-progress[data-mobile-open="false"]' in css.text
    assert '#mobile-step-toggle[aria-expanded="true"]' in css.text
    assert "#6366f1" not in css.text


def test_setup_home_renders_shell_and_collapsed_detail_drawers() -> None:
    home = client.get("/")
    workbench_classes = class_tokens_for_id(home.text, "setup-workbench")
    progress_classes = class_tokens_for_id(home.text, "setup-progress")
    prerequisites_panel_classes = class_tokens_for_id(
        home.text, "step-panel-prerequisites"
    )
    configuration_panel_classes = class_tokens_for_id(
        home.text, "step-panel-configuration"
    )
    runtime_panel_classes = class_tokens_for_id(home.text, "step-panel-runtime")
    process_panel_classes = class_tokens_for_id(home.text, "step-panel-process")

    assert home.status_code == 200
    assert 'id="setup-workbench"' in home.text
    assert 'id="setup-progress"' in home.text
    assert 'id="readiness-summary"' in home.text
    assert 'id="blocker-summary"' in home.text
    assert 'id="common-fixes"' in home.text
    assert 'data-step-id="prerequisites"' in home.text
    assert 'data-step-id="configuration"' in home.text
    assert 'data-step-id="runtime"' in home.text
    assert 'data-step-id="process"' in home.text
    assert "shell" in workbench_classes
    assert "progress-rail" in progress_classes
    assert {"step-panel", "card"} <= prerequisites_panel_classes
    assert {"step-panel", "card", "wide"} <= configuration_panel_classes
    assert {"step-panel", "card"} <= runtime_panel_classes
    assert {"step-panel", "card"} <= process_panel_classes
    assert 'id="mobile-step-toggle"' in home.text
    assert 'aria-controls="setup-progress"' in home.text
    assert 'aria-expanded="false"' in home.text
    assert 'id="run-diagnostics"' in home.text
    assert 'id="copy-env"' in home.text
    assert 'id="runtime-check"' in home.text
    assert 'id="bootstrap-models"' in home.text
    assert 'id="process-run"' in home.text
    assert 'id="runtime-summary"' in home.text
    assert 'id="process-summary"' in home.text
    assert 'id="advanced-config"' in home.text
    assert 'id="runtime-details"' in home.text
    assert 'id="process-details"' in home.text
    assert re.search(
        r'<details[^>]*id="advanced-config"(?![^>]*\bopen\b)[^>]*>', home.text
    )
    assert re.search(
        r'<details[^>]*id="runtime-details"(?![^>]*\bopen\b)[^>]*>', home.text
    )
    assert re.search(
        r'<details[^>]*id="process-details"(?![^>]*\bopen\b)[^>]*>', home.text
    )


def test_setup_assets_expose_module_and_top_level_contracts() -> None:
    home = client.get("/")
    entry_js = client.get("/static/setup.js")
    api_js = client.get("/static/setup_api.js")
    state_js = client.get("/static/setup_state.js")
    view_models_js = client.get("/static/setup_view_models.js")
    views_js = client.get("/static/setup_views.js")

    assert 'src="static/setup.js" type="module"' in home.text

    for response in (entry_js, api_js, state_js, view_models_js, views_js):
        assert response.status_code == 200
        assert _is_javascript_content_type(response.headers["content-type"])

    assert 'import { createSetupApi } from "./setup_api.js";' in entry_js.text
    assert 'import { createSetupState } from "./setup_state.js";' in entry_js.text
    assert (
        'import { buildShellSummaryView } from "./setup_view_models.js";'
        in entry_js.text
    )
    assert (
        'import { buildProcessSummaryView } from "./setup_view_models.js";'
        in entry_js.text
    )
    assert (
        'import { buildProcessFailureView } from "./setup_view_models.js";'
        in entry_js.text
    )
    assert 'import { createSetupViews } from "./setup_views.js";' in entry_js.text
    assert "function boot()" in entry_js.text
    assert "boot();" in entry_js.text
    assert "handleDiagnosticsRun" in entry_js.text
    assert "handleRuntimeCheck" in entry_js.text
    assert "handleBootstrapModels" in entry_js.text
    assert "handleProcessRun" in entry_js.text
    assert "runDiagnostics();" not in entry_js.text

    assert "export function createSetupApi" in api_js.text
    assert "runDiagnostics" in api_js.text
    assert "checkRuntime" in api_js.text
    assert "bootstrapModels" in api_js.text
    assert "runProcess" in api_js.text

    assert "export function createSetupState" in state_js.text
    assert "beginDiagnosticsCheck" in state_js.text
    assert "markConfigurationComplete" in state_js.text
    assert "beginRuntimeCheck" in state_js.text
    assert "completeRuntimeCheck" in state_js.text
    assert "beginProcess" in state_js.text
    assert "markProcessSuccess" in state_js.text
    assert "markProcessFailure" in state_js.text
    assert (
        'const STEP_ORDER = ["prerequisites", "configuration", "runtime", "process"]'
        in state_js.text
    )

    assert "export function buildShellSummaryView" in view_models_js.text
    assert "export function buildStaleSummaryLabel" in view_models_js.text
    assert "export function buildProcessSummaryView" in view_models_js.text
    assert "export function buildProcessFailureView" in view_models_js.text

    assert "export function createSetupViews" in views_js.text
    assert "renderShellState" in views_js.text
    assert "renderDiagnosticsSummary" in views_js.text
    assert "renderRuntimeSummary" in views_js.text
    assert "renderRuntimeDetails" in views_js.text
    assert "renderProcessSummary" in views_js.text
    assert "renderProcessDetails" in views_js.text
    assert "bindMobileStepToggle" in views_js.text
    assert 'document.getElementById("mobile-step-toggle")' in views_js.text
    assert (
        'progressRail.dataset.mobileOpen = expanded ? "true" : "false";'
        in views_js.text
    )
    assert (
        'mobileStepToggle.setAttribute("aria-expanded", String(expanded));'
        in views_js.text
    )
    assert "views.bindMobileStepToggle();" in entry_js.text


def test_setup_js_prefers_backend_setup_views_for_state_transitions() -> None:
    entry_js = client.get("/static/setup.js")

    assert entry_js.status_code == 200
    assert "function toDiagnosticsView(report)" in entry_js.text
    assert (
        'return report.setup_view || { state: report.ready ? "ready" : "blocked" };'
        in entry_js.text
    )
    assert "function toRuntimeView(runtime)" in entry_js.text
    assert (
        'return runtime.setup_view || { state: runtime.reachable ? "ready" : "blocked" };'
        in entry_js.text
    )
    assert "const diagnosticsView = toDiagnosticsView(report);" in entry_js.text
    assert "state.applyDiagnosticsResult(diagnosticsView);" in entry_js.text
    assert "const runtimeView = toRuntimeView(runtime);" in entry_js.text
    assert "state.completeRuntimeCheck(runtimeView);" in entry_js.text


def test_setup_js_renders_shaped_setup_view_summaries_and_common_fixes() -> None:
    entry_js = client.get("/static/setup.js")

    assert entry_js.status_code == 200
    assert "const diagnosticsView = toDiagnosticsView(report);" in entry_js.text
    assert "views.renderDiagnosticsSummary(diagnosticsView);" in entry_js.text
    assert "views.renderCommonFixes(diagnosticsView);" in entry_js.text
    assert 'if (diagnosticsView.state === "ready") {' in entry_js.text
    assert "if (report.ready) {" not in entry_js.text
    assert "const runtimeView = toRuntimeView(runtime);" in entry_js.text
    assert "renderRuntimeState(runtimeView, runtime);" in entry_js.text
    assert (
        "renderRuntimeState(runtimeView, { ...runtime, bootstrap: report });"
        in entry_js.text
    )


def test_setup_views_expose_common_fixes_and_runtime_readiness_copy() -> None:
    views_js = client.get("/static/setup_views.js")

    assert views_js.status_code == 200
    assert "function renderCommonFixes(view)" in views_js.text
    assert 'const container = document.getElementById("common-fixes");' in views_js.text
    assert "const fixes = Array.isArray(view?.checks)" in views_js.text
    assert "view.checks.filter((check) => check.fix)" in views_js.text
    assert "function renderRuntimeSummary(runtimeView, runtime)" in views_js.text
    assert (
        '["Required models", `${requiredModels.length} configured`],' in views_js.text
    )
    assert '["Available locally", `${localModels.length} ready`],' in views_js.text
    assert "runtimeView?.next_action ||" in views_js.text
    assert "renderCommonFixes," in views_js.text


def test_setup_process_summary_view_reads_nested_model_used() -> None:
    view_models_js = client.get("/static/setup_view_models.js")

    assert view_models_js.status_code == 200
    assert "function collectProcessModel(result)" in view_models_js.text
    assert "result.summary?.model_used" in view_models_js.text
