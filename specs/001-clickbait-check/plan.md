# Implementation Plan: Clickbait Headline Check

**Branch**: `001-clickbait-check` | **Date**: 2025-12-17 | **Spec**: specs/001-clickbait-check/spec.md  
**Input**: Feature specification from `/specs/001-clickbait-check/spec.md`

## Summary

Expose a Python clickbait detector (existing `code/klikbait/predict.py` + model assets) as a FastAPI endpoint and surface the result beside each headline in the frontend. Keep inference in Python with a documented contract, reuse current backend app structure, and add a lightweight UI indicator without duplicating logic in Next.js.

## Technical Context

**Language/Version**: Python 3.11 (backend), TypeScript/Next.js (frontend)  
**Primary Dependencies**: FastAPI + Pydantic; Hugging Face transformers/torch for clickbait model; Next.js App Router for UI  
**Storage**: None (in-memory only)  
**Testing**: pytest for backend; frontend relies on existing Next.js testing setup (assumed Playwright/React Testing Library)  
**Target Platform**: Linux local dev; frontend served via Next.js dev server  
**Project Type**: Web app with Python backend + Next.js frontend  
**Performance Goals**: Clickbait label available within ~1s for typical headline length; non-blocking page render  
**Constraints**: Offline/local execution; deterministic model version; avoid adding cloud dependencies; keep endpoint CPU-friendly  
**Scale/Scope**: Single-page headline enrichment; internal API consumption by existing frontend  
**Unknowns**: `train_clickbait.py` not present; scope set to inference-only with existing model unless retraining is explicitly requested later.

## Constitution Check

- **Python as Source of Truth**: Clickbait inference stays in Python (existing model + predict wrapper). ✔️
- **Explicit Interface**: New HTTP contract documented in `/specs/001-clickbait-check/contracts/`; versioned fields included. ✔️
- **Single Responsibility**: Clickbait detector isolated in backend service layer; frontend only renders labels. ✔️
- **Frontend as Thin Client**: UI reads API response and renders badges; no model logic client-side. ✔️
- **Deterministic Execution**: Fixed model path/version and deterministic pipeline notes in quickstart/contracts. ✔️

## Project Structure

### Documentation (this feature)

```text
specs/001-clickbait-check/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── checklists/
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/          # FastAPI app, routers, schemas
│   ├── services/     # analyzer + new clickbait adapter
│   ├── lib/          # determinism helpers
│   └── models/       # shared types (existing)
└── tests/

code/klikbait/        # Clickbait model assets and predict wrapper

frontend/
├── app/              # Next.js app router pages/layout
└── components/       # UI components (labels/badges)
```

**Structure Decision**: Use existing backend FastAPI project plus Next.js frontend; add clickbait router/service in backend and UI indicator component in frontend.

## Complexity Tracking

> No constitutional violations; table remains empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|

## Phase 0: Outline & Research

**Unknowns to resolve**
- Confirm expectation for `train_clickbait.py` (not present); assume inference-only with `predict.py` and bundled model unless user requests retraining.

**Best-practice scans**
- FastAPI endpoint design for lightweight ML inference (validation, timeouts, deterministic responses).
- Next.js rendering pattern for per-headline badges without blocking page render.

**Integration patterns**
- Endpoint path: POST `/clickbait/analyze` via new router to avoid changing existing `/analyze` payload shape while keeping shared app.
- Response payload: `is_clickbait` (bool), `score` (0–1), `label` (string), optional `confidence_note`, `version`/`contract_version` fields, and `errors` array on failure.

**Research output**
- Documented in `specs/001-clickbait-check/research.md` with decisions, rationale, and alternatives. All clarifications resolved with inference-only assumption noted.

## Phase 1: Design & Contracts

- Data model recorded in `specs/001-clickbait-check/data-model.md` (Headline, ClickbaitEvaluation with validation).
- API contract added in `specs/001-clickbait-check/contracts/clickbait-analyze.yaml` for POST `/clickbait/analyze`.
- Quickstart in `specs/001-clickbait-check/quickstart.md` covering backend/ frontend steps and model location.
- Agent context refreshed via `.specify/scripts/bash/update-agent-context.sh codex`.
- Constitution Check (post-design): PASS — Python owns inference; explicit contract documented; frontend remains thin; deterministic model/version noted.

## Phase 2: Implementation Plan (handoff to `/speckit.tasks`)

- Backend: add FastAPI router/service wrapping `code/klikbait/predict.py`, request/response schemas, validation, error handling, determinism metadata, and tests.
- Frontend: call new endpoint per headline (or batch if available), render badge with label/score, handle loading/error/timeout with accessible text.
- Ops/dev: ensure model assets available locally, document env vars/paths, wire into app startup and health checks as needed.
