# Implementation Plan: News Analysis MVP

**Branch**: `001-news-analysis-mvp` | **Date**: 2025-12-14 | **Spec**: specs/001-news-analysis-mvp/spec.md
**Input**: Feature specification from `/specs/001-news-analysis-mvp/spec.md`

**Note**: Complete the Constitution Check before proceeding to research or design.

## Summary

Python backend owns parsing and sentiment: it accepts a news URL or raw text, extracts title/author/
date/text, runs RuBERT sentiment deterministically, and returns JSON with model/contract versions and
seeds. Next.js is a thin UI that only submits input and renders the backend JSON. Interaction is a
single HTTP POST `/analyze` callable locally on CPU-only Linux. No cloud, auth, or deployment work.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11; Next.js (App Router)  
**Primary Dependencies**: FastAPI, uvicorn, requests, BeautifulSoup/newspaper3k (parsing), torch +
transformers (RuBERT), pydantic  
**Storage**: None (in-memory, no persistence)  
**Testing**: pytest for backend; React Testing Library optional for UI snapshots  
**Target Platform**: Linux (CPU-only), local execution  
**Project Type**: Web (backend + frontend)  
**Performance Goals**: <=5s response for typical news article on CPU-only laptop; identical outputs
for identical inputs  
**Constraints**: No cloud, no auth, no deployment; deterministic seeds; Python owns all logic; thin
frontend; offline/local friendly  
**Scale/Scope**: Academic MVP single-user, single-tenant; extensible for future analyzers

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Python as Source of Truth**: Plan keeps parsing/sentiment/data-extraction logic in Python,
  runnable standalone (CLI/HTTP) without the frontend.
- **Explicit Interface**: Request/response schemas for Python ↔ Next.js are documented, versioned,
  and committed before integration (HTTP or CLI only).
- **Single Responsibility**: Modules do not mix unrelated tasks; any justified deviation is added to
  the Complexity Tracking table.
- **Frontend as Thin Client**: Frontend only collects input, orchestrates calls, and renders output;
  no inference or parsing logic is planned in Next.js.
- **Deterministic Execution**: Model versions/seeds and preprocessing steps are pinned; plan confirms
  local execution without cloud dependencies.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
```text
backend/
├── src/
│   ├── api/              # FastAPI app + request/response schemas
│   ├── services/         # parsing, sentiment orchestration
│   ├── models/           # domain entities
│   └── cli/              # optional local CLI wrapper
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
└── app/                  # Next.js App Router pages/components
    ├── api/              # client fetch helpers
    └── components/
```

**Structure Decision**: Two-project web layout (backend, frontend). Backend holds all parsing/ML
logic; frontend only orchestrates and renders JSON responses.

## Implementation Sequence (incremental)

1) **Scaffold backend**: create FastAPI app, pydantic schemas, deterministic seed setup, health ping.  
2) **Implement parsing service**: URL fetch with requests + BeautifulSoup/newspaper3k; handle missing
   metadata gracefully; normalize dates to ISO.  
3) **Wire sentiment service**: load fixed RuBERT model weights (CPU), pin seeds, return label + score
   only; expose contract_version and model_version.  
4) **Expose /analyze endpoint**: accept url|text input, validate, orchestrate parsing + sentiment,
   return structured JSON with reproducibility metadata and error codes.  
5) **Backend tests**: unit tests for parsing and sentiment determinism; contract test for /analyze
   happy/invalid paths; integration test end-to-end CPU run.  
6) **Frontend thin UI**: single page with URL/text input, submit to backend HTTP, render JSON fields
   verbatim; basic error display; no client-side heuristics.  
7) **Frontend checks**: simple UI render test/snapshot; manual run against local backend.  
8) **Determinism validation**: run identical input multiple times, confirm identical outputs and
   logged seeds/model_version.  
9) **Docs**: update quickstart with local run steps; ensure contracts and data-model checked in.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
