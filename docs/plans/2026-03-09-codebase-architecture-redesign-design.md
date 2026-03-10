# Codebase Architecture Redesign

Date: 2026-03-09
Status: Approved

## Goal

Define an ideal end-state architecture for the Video RSS Aggregator that improves
maintainability, reliability, delivery velocity, and testability.

## Context

The current codebase works, but several design pressures now slow it down:

- `Pipeline` mixes composition, orchestration, feed ingestion, processing,
  persistence, runtime reporting, and RSS generation.
- Data contracts leak across layers, especially between summarization,
  persistence, and RSS rendering.
- CLI and API do not consistently share one application-level workflow model.
- Setup/runtime data is shaped in multiple places across Python, HTML, and JS.
- Tests rely heavily on monkeypatching concrete modules instead of stable seams.

## Chosen Approach

Adopt a ports-and-use-cases architecture with four layers:

1. Adapters
2. Application
3. Domain
4. Infrastructure

This was selected over a lighter workflow-slice refactor or a stabilized version
of the current layering because the goal is an ideal long-term architecture,
not only a lower-risk cleanup.

## Architectural Principles

- Dependencies point inward only.
- Business workflows live in application use cases, not transport adapters.
- Domain types are stable and independent of storage, HTTP, CLI, and model APIs.
- Infrastructure implements ports instead of owning business rules.
- Composition happens once in a single composition root.

## Target Architecture

### Adapters

Adapters translate external interactions into application requests and map
application responses back out.

- FastAPI routes
- CLI commands
- GUI/setup endpoints and page models
- RSS HTTP delivery surface

Adapters should not contain orchestration logic beyond request mapping,
validation, and response formatting.

### Application

Application use cases coordinate business workflows through explicit ports.

Core use cases:

- `BootstrapRuntime`
- `GetRuntimeStatus`
- `IngestFeed`
- `ProcessSource`
- `RenderRssFeed`

The current `Pipeline` class should be replaced by these focused use cases.

### Domain

Domain types define the stable language of the system.

Illustrative types:

- `SourceItem`
- `VideoRecord`
- `Transcript`
- `PreparedMedia`
- `SummaryDraft`
- `SummaryResult`
- `ProcessOutcome`
- `DiagnosticReport`

These types must not depend on SQLite rows, Ollama payloads, FastAPI models,
Click commands, or subprocess output.

### Infrastructure

Infrastructure adapters implement application ports.

- SQLite repositories
- Ollama client adapter
- Feed fetching adapter
- Media preparation adapter around yt-dlp, ffmpeg, and filesystem artifacts
- RSS rendering adapter
- Runtime inspection adapter

Infrastructure owns transport and tool integration details, but not workflow
decisions.

## Ports

The application layer should depend on explicit interfaces such as:

- `FeedSource`
- `VideoRepository`
- `SummaryRepository`
- `MediaPreparationService`
- `Summarizer`
- `RuntimeInspector`
- `PublicationRenderer`
- `ArtifactStore`

This creates narrow seams for tests and keeps adapters replaceable.

## Data Flow

### Startup

The composition root loads `Config`, builds infrastructure adapters, wires use
cases, and exposes them to FastAPI and CLI entry points. Web startup and
shutdown should be owned by FastAPI lifespan rather than by a prebuilt runtime
object created externally.

### Ingest

`IngestFeed` fetches and parses a feed, normalizes entries into domain types,
stores feed/video metadata, and optionally delegates processing to
`ProcessSource`. It should not own processing internals.

### Processing

`ProcessSource` asks `MediaPreparationService` for `PreparedMedia`, passes the
result to `Summarizer`, then persists a typed `ProcessOutcome`.

This use case owns the decision about whether a result is successful, degraded,
or failed.

### Publication

`RenderRssFeed` reads published summaries through repositories and passes stable
publication models to a renderer. RSS generation should not depend on storage
row types.

### Setup and Runtime

`BootstrapRuntime` and `GetRuntimeStatus` should return one application-level
view model shared by API and GUI, replacing duplicated config/setup shaping.

## Error Handling Model

Replace implicit fallback-heavy success semantics with explicit outcome types:

- `Success`
- `PartialSuccess`
- `Failure`

Rules:

- Adapter-specific exceptions are translated at the use-case boundary.
- `PartialSuccess` is used when the system produced a degraded but valid result.
- `Failure` means the business goal was not achieved and must not be presented
  as a normal success.
- Persistence records outcome status explicitly rather than relying on summary
  text to reveal degradation.
- API and CLI surfaces report status directly.

Diagnostics remain separate from processing outcomes.

## Testing Strategy

The main regression net should move to application-boundary contract tests for:

- `BootstrapRuntime`
- `GetRuntimeStatus`
- `IngestFeed`
- `ProcessSource`
- `RenderRssFeed`

Supporting tests should be split into:

- repository integration tests
- Ollama adapter tests
- media adapter tests
- FastAPI adapter tests
- CLI adapter tests
- policy tests for model selection, degradation classification, normalization,
  and retention/publication rules

The desired end state is that most business behavior can be tested without
FastAPI, Click, SQLite, subprocesses, or a live Ollama runtime.

## Non-Goals

- Defining the full migration sequence in this document
- Implementing the redesign directly from adapters inward
- Preserving the current `Pipeline` shape as a compatibility constraint

## Expected Benefits

- clearer ownership of business workflows
- lower coupling between persistence, summarization, and presentation
- consistent behavior across API and CLI
- easier unit and contract testing
- safer future changes to model policy, media tooling, and publishing

## Next Step

Create a dedicated implementation plan that stages the migration from the
current codebase into this target architecture.
