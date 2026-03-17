# Setup Studio UX Redesign

Date: 2026-03-17
Status: Approved

## Goal

Redesign the setup studio so first-time Windows users can reach a correct,
working local setup with fewer mistakes and less confusion.

The primary optimization target is not speed. It is error prevention:

- make the next safe action obvious
- surface blockers before deeper configuration work
- reduce overwhelming walls of fields and raw output
- preserve access to advanced controls without making them the default path

## Context

The current setup studio already exposes the right backend capabilities, but the
frontend shape makes them harder to use safely:

- the page presents prerequisites, full configuration, runtime checks, and a
  first processing run with roughly equal weight on one screen
- diagnostics auto-run on page load, which creates output before the user has
  context for what they are seeing
- configuration is shown as a large all-fields form instead of a guided,
  beginner-safe workflow
- raw JSON and log-like output dominate the feedback model
- the current dark control-panel styling feels more like an expert dashboard
  than a first-run setup companion

The backend seams already align well with a guided setup flow:

- `GET /setup/config`
- `GET /setup/diagnostics`
- `POST /setup/bootstrap`
- `GET /runtime`
- `POST /process`

The redesign should reuse these surfaces instead of requiring a backend rewrite.

## Chosen Approach

Adopt a guided workbench layout with persistent progress, step-level guardrails,
and progressive disclosure.

This approach was chosen over:

1. a strict wizard, which would reduce mistakes well but add too much friction
   for repeated use
2. a denser mission-control dashboard, which would preserve speed but keep too
   much cognitive load for first-time users

The guided workbench keeps the user oriented with a visible checklist while
focusing the main panel on one step at a time.

## UX Principles

- Beginner-first by default, expert-capable on demand
- Explain what matters before exposing every tuning knob
- Make status human-readable first, machine-readable second
- Prevent mistakes earlier instead of describing them after failure
- Keep progress visible and reversible
- Use an instructional tone instead of a dashboard tone

## Information Architecture

### Overall Layout

Desktop layout uses two coordinated regions:

1. a left-side progress rail
2. a right-side active step panel

The left rail contains:

- overall readiness summary
- current blocker summary
- four-step checklist with state indicators
- quick help and common-fix links

The right panel contains the currently active step only:

- step title and short explanation
- why the step matters
- required checks or inputs
- primary action area
- result summary and next action
- advanced details behind collapsible sections

Mobile layout collapses this into a current-step-first view with progress and
the full checklist available behind an expander or sheet.

Mobile uses responsive reflow, not a different product flow. The checklist,
blocker summaries, advanced details, and actions remain the same concepts with a
different presentation.

### Step Order

1. Install prerequisites
2. Build configuration
3. Verify runtime
4. Run first processing test

This order matches the real dependency chain and prevents users from editing
deep runtime knobs before the environment is viable.

### Progress Persistence

The workbench does not introduce a durable server-side wizard session.

- progress is a client-side view model
- configuration fields rehydrate from `GET /setup/config` on load
- prerequisite, runtime, and processing status are derived from the latest check
  results collected in the current browser session
- after reload, steps return to an unverified state unless they can be derived
  directly from current config payloads

This keeps the UI honest about what has been verified while avoiding new backend
state-management scope.

## Step Design

### 1. Install Prerequisites

Primary source: `GET /setup/diagnostics`

Purpose:

- confirm Python, FFmpeg, FFprobe, yt-dlp, and Ollama availability
- show exact missing items before any deeper setup work

Default presentation:

- dependency cards with clear pass/fail state
- short explanation of why each dependency matters
- specific recovery guidance for failed checks
- explicit `Run checks` or `Re-check` action

Not shown by default:

- raw diagnostic payloads
- large unstructured output blocks

### 2. Build Configuration

Primary source: `GET /setup/config`

Purpose:

- let the user produce a correct starter configuration without facing the full
  tuning surface immediately

Default presentation:

- basic mode for the small set of fields that matter first
- advanced mode for VRAM budgeting, thresholds, retention, and similar tuning
- generated `.env` preview as an output artifact
- copy actions for `.env` and quick commands

Basic mode should prioritize:

- bind address
- storage directory
- Ollama URL
- model priority

RSS title, link, and description should live in advanced mode with their current
defaults preserved. They are useful for publishing setup, but not necessary for
first-run validation.

Advanced mode should contain:

- budget ratio
- reserve amount
- frame extraction thresholds
- transcript and summary retention
- token and output tuning

The full `.env` block remains available, but it should no longer be the primary
editing model.

### 3. Verify Runtime

Primary sources:

- `GET /runtime`
- `POST /setup/bootstrap`

Purpose:

- prove the configured runtime is reachable and ready to summarize

Default presentation:

- connection status in plain language
- whether Ollama is reachable
- visible local-model readiness
- missing-model explanation before bootstrap
- bootstrap progress and completion state

Advanced runtime payload details can remain accessible through a disclosure, but
the first screen should answer a simple question: can this system actually run?

### 4. Run First Processing Test

Primary source: `POST /process`

Purpose:

- give the user a confidence-building proof that the full pipeline works end to
  end

Default presentation:

- sample source prefilled with the current public default already used by the UI:
  `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- optional title override
- expectation setting that processing may take time
- a success summary emphasizing outcome, not raw response shape

The sample source remains editable. If the default source is unavailable or the
request fails, the UI keeps the step in a failed state, preserves the entered
value, and suggests retrying with a different URL or local file path.

The first success view should highlight:

- processed title
- transcript size
- frame count
- model used
- generated summary highlights

Raw response details should remain available behind a details section.

## Guardrail Model

Each step uses a small set of explicit states:

- `unverified`
- `blocked`
- `ready`
- `running`
- `failed`
- `complete`

Rules:

- diagnostics do not auto-run on initial page load
- page load initializes prerequisite, runtime, and processing steps as
  `unverified` until the user runs them
- steps that depend on unmet prerequisites stay visibly blocked
- blocked steps explain both the problem and the exact next fix
- failed steps preserve the most recent meaningful result, show recovery
  actions, and allow retry without clearing user-entered inputs
- completing one step highlights the recommended next step
- completed steps remain editable without losing overall orientation
- advanced controls are available but collapsed by default

This model changes the interface from an output-heavy control panel into a
guided state machine.

## Frontend Units and Boundaries

The implementation plan should decompose the setup studio into a small set of
clear frontend units even if the first pass remains in the existing template,
CSS, and JS files.

Suggested units:

- `setup shell`: template structure for the progress rail, active step panel,
  helper sections, and mobile variants
- `step state controller`: client-side state machine that owns transitions,
  gating rules, and current-step selection
- `setup API client`: thin wrappers around the existing setup/runtime/process
  endpoints
- `view-model mappers`: functions that convert raw diagnostics, runtime, and
  process payloads into human-readable summaries and blocker/fix models
- `step renderers`: focused DOM update logic for each step's summary, actions,
  and detail sections
- `adapter regression tests`: tests that verify structure, state-driven
  behavior, and critical content without depending on visual styling details

Boundaries:

- API client functions should not decide UX copy or state transitions
- view-model mappers should not manipulate the DOM directly
- the state controller should consume mapped results and decide whether a step
  is unverified, blocked, ready, failed, running, or complete
- DOM renderers should be replaceable without changing endpoint integration

This gives the planner independently understandable and testable units without
forcing an unnecessary framework migration.

## Content and Messaging Strategy

Every step should answer the same questions in the same order:

1. what are we checking or configuring?
2. why does it matter?
3. what should the user do now?
4. what happened?
5. what comes next?

Messaging style should be:

- direct
- plain-language
- specific about recovery steps
- calm rather than alarmist
- instructional rather than overly technical

Raw JSON, full diagnostic objects, and other machine-shaped data should move to
secondary detail views.

## Visual Direction

Use the approved `Workshop Manual` direction.

Characteristics:

- warm neutral palette rather than dark glassmorphism
- supportive, trust-building visual tone
- stronger reading comfort for first-run guidance
- status colors used sparingly and intentionally
- interface that feels instructional and grounded instead of like a mission
  control board

This direction should replace the current heavy dark background and purple-led
accent language in `video_rss_aggregator/static/setup.css`.

## Data and Interaction Mapping

The redesign should preserve the current backend contract and remap the frontend
around it.

- load configuration defaults from `GET /setup/config`
- run dependency checks only when the user starts or repeats them via
  `GET /setup/diagnostics`
- check runtime readiness via `GET /runtime`
- bootstrap models via `POST /setup/bootstrap`
- run the first validation job via `POST /process`

Allowed backend changes are intentionally narrow:

- no new workflow endpoints for the redesign's first implementation pass
- optional response-shaping additions to existing setup/runtime endpoints if the
  frontend needs clearer view-model fields
- no changes to application-use-case responsibilities beyond exposing clearer
  adapter-facing payloads

If response shaping is needed, it should happen at the FastAPI adapter boundary
or in lightweight view-model helpers, not by expanding core domain or workflow
scope.

## Error Handling

- Show blocker summaries inline at the step level
- Pair failures with actionable fixes, not just error text
- Keep raw error details available in expandable sections
- Avoid presenting partial or failed operations as generic success states
- Preserve user-entered configuration while retrying checks or actions
- Keep the last successful summary visible when a later retry fails, clearly
  labeled as stale if needed

## Testing Strategy

Update the setup studio regression net to validate the new guided experience.

Key coverage areas:

- setup page structure reflects the step-based workbench
- beginner-first content appears before advanced controls
- diagnostics are not auto-triggered on page load
- `.env` generation still reflects edited configuration values
- runtime and processing results render human-readable summaries
- advanced details remain available for debugging without dominating the default
  view

Tests should continue to protect the existing setup/runtime/process API
contracts, while adapter-facing UI tests evolve to reflect the new structure.

## Non-Goals

- redesigning the underlying runtime architecture
- adding new core setup endpoints unless view-model gaps make them necessary
- building a multi-page onboarding flow disconnected from the current server UI
- optimizing primarily for expert repeat use at the expense of first-run safety

## Expected Benefits

- fewer first-run setup mistakes
- clearer progression from install to successful processing
- less intimidation from the configuration surface
- improved trust in runtime readiness and failure recovery
- a more distinctive and intentional setup experience

## Next Step

Create an implementation plan for the setup studio redesign, staged around
template structure, interaction model, visual system, and regression updates.
