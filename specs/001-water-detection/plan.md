# Implementation Plan: Water Content Detection

**Branch**: `001-water-detection` | **Date**: 2025-12-22 | **Spec**: /home/termenater/vibe/specs/001-water-detection/spec.md
**Input**: Feature specification from `/specs/001-water-detection/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. Complete the Constitution Check
before proceeding to research or design.

## Summary

Expose the existing `code/water/water_analyzer.py` model via a documented HTTP endpoint and surface water/not-water labels with confidence and feature interpretations in the frontend. Keep all detection logic in Python, provide deterministic request/response contracts, and render results inline in the site with graceful fallbacks for invalid input or backend issues.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (backend), TypeScript/Next.js App Router (frontend)  
**Primary Dependencies**: FastAPI/uvicorn for HTTP layer, WaterAnalyzer (joblib + pandas + pymorphy3 + nltk), requests/BS4/newspaper3k already present for other features  
**Storage**: None (in-memory only)  
**Testing**: pytest for backend contracts; frontend smoke/integration tests via existing Next.js test setup (assumed)  
**Target Platform**: Linux dev/server environment; local-first, no cloud dependencies  
**Project Type**: Web (backend + frontend)  
**Performance Goals**: 95% of valid submissions labeled in UI within 2 seconds; API success ≥98% on first attempt in sample runs  
**Constraints**: Deterministic outputs with fixed model artifacts; offline-capable; UI must stay responsive with fallbacks on timeout/failure  
**Scale/Scope**: Single feature across existing backend/frontend; per-request text up to ~10k chars; sequential evaluations per page session

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Python as Source of Truth**: PASS — water detection logic remains in `code/water/water_analyzer.py`; backend exposes it; frontend only consumes results.
- **Explicit Interface**: PASS — plan includes documented HTTP contract for `/api/water-detection` with version tokens before integration.
- **Single Responsibility**: PASS — analyzer remains in its module; backend endpoint handles transport/validation; frontend handles rendering only.
- **Frontend as Thin Client**: PASS — no ML or heuristic logic on frontend; only displays labels and interpretations.
- **Deterministic Execution**: PASS — fixed model artifact + deterministic preprocessing; responses versioned; local execution only.

**Post-Phase-1 Re-check**: PASS — design artifacts (data model, contracts, quickstart) keep logic in Python with explicit HTTP contract and deterministic behavior.

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
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
backend/
├── src/                # existing backend code and APIs
├── tests/              # backend tests (extend with contract/integration for water detection)

frontend/
├── src/                # Next.js App Router frontend
└── tests/              # frontend tests (smoke/integration for UI rendering)

code/
└── water/
    └── water_analyzer.py  # existing detection logic and model loading

specs/
└── 001-water-detection/   # spec, plan, research, contracts, quickstart, data-model
```

**Structure Decision**: Use existing backend/frontend separation with `code/water` as the Python model source; add contracts/docs under `specs/001-water-detection` and new backend endpoint plus frontend integration within current folders.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| _None_ | _N/A_ | _N/A_ |
