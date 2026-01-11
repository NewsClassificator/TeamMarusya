---

description: "Task list template for feature implementation"
---

# Tasks: Water Content Detection

**Input**: Design documents from `/specs/001-water-detection/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are optional and not explicitly requested in the spec; tasks focus on implementation and observable acceptance behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure dependencies and paths for the water detector are available to backend/frontend.

- [X] T001 Update Python dependencies to include `joblib` and `pymorphy3` in `/home/termenater/vibe/backend/pyproject.toml`.
- [X] T002 Document/default water model artifact and module locations in `/home/termenater/vibe/backend/README.md` for team setup.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core backend plumbing required before user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 Create water detector config with contract/detector versions, model path, module path, and text bounds in `/home/termenater/vibe/backend/src/lib/water_config.py`.
- [X] T004 Add WaterAnalyzer loader and deterministic context setup in `/home/termenater/vibe/backend/src/services/water_detector.py` (module import, model load, basic feature call scaffold).
- [X] T005 Wire FastAPI app to expect a water router import (placeholder include) in `/home/termenater/vibe/backend/src/api/app.py`.

**Checkpoint**: Foundation ready – backend can load WaterAnalyzer and is ready to expose an endpoint.

---

## Phase 3: User Story 1 - See water label on text (Priority: P1) 🎯 MVP

**Goal**: Show water/not-water label with confidence and feature interpretations in the frontend.

**Independent Test**: Submit text via UI and see label + confidence (with on-demand details) without relying on other new stories beyond the water endpoint.

### Implementation for User Story 1

- [X] T006 [US1] Add Next.js proxy route for water detection at `/home/termenater/vibe/frontend/app/api/water-detection/route.ts` forwarding POST to backend `/api/water-detection` and surfacing backend status.
- [X] T007 [P] [US1] Create water badge/insight component to display label, confidence, and interpretations in `/home/termenater/vibe/frontend/app/components/WaterBadge.tsx`.
- [X] T008 [US1] Update UI flow to request water detection on submitted text and manage loading/error/cache state in `/home/termenater/vibe/frontend/app/components/AnalyzePageClient.tsx`.
- [X] T009 [US1] Render water detection results inline with article content (label + confidence, optional interpretations toggle) in `/home/termenater/vibe/frontend/app/components/AnalysisResult.tsx`.

**Checkpoint**: UI shows water/not-water status with confidence and explanations for a submitted text.

---

## Phase 4: User Story 2 - Request water check via API (Priority: P2)

**Goal**: Provide HTTP API that returns water decision, confidence, and feature metrics.

**Independent Test**: Call API with valid text and receive deterministic response containing decision, confidence, and feature metrics per contract.

### Implementation for User Story 2

- [X] T010 [US2] Define request/response schemas with validation (text 20–10,000 chars, include_features default true) in `/home/termenater/vibe/backend/src/api/schemas_water.py`.
- [X] T011 [US2] Implement water detection service to run `WaterAnalyzer`, map features/interpretations, and set version fields in `/home/termenater/vibe/backend/src/services/water_detector.py`.
- [X] T012 [US2] Add FastAPI route for POST `/api/water-detection` using schemas and service in `/home/termenater/vibe/backend/src/api/routes_water.py`.
- [X] T013 [US2] Register water router in the application factory so endpoint is exposed in `/home/termenater/vibe/backend/src/api/app.py`.
- [X] T014 [US2] Align Next proxy target/base URL and error passthrough for water detection in `/home/termenater/vibe/frontend/app/api/water-detection/route.ts`.

**Checkpoint**: API responds with structured water detection output per contract, callable independently of UI.

---

## Phase 5: User Story 3 - Graceful handling of invalid or failed checks (Priority: P3)

**Goal**: Reject invalid input and degrade gracefully on detector failures/timeouts while keeping the page usable.

**Independent Test**: Send invalid or failing requests; UI shows clear feedback/fallback while remaining usable.

### Implementation for User Story 3

- [X] T015 [US3] Enforce validation and structured error responses (empty/short/oversized text, unsupported characters) in `/home/termenater/vibe/backend/src/api/schemas_water.py` and service fallbacks in `/home/termenater/vibe/backend/src/services/water_detector.py`.
- [X] T016 [US3] Add backend neutral fallback path on detector timeout/error that returns status-unavailable label in `/home/termenater/vibe/backend/src/services/water_detector.py`.
- [X] T017 [US3] Handle frontend fallback states (display neutral message, avoid blocking other content) when water detection errors or times out in `/home/termenater/vibe/frontend/app/components/AnalyzePageClient.tsx` and `/home/termenater/vibe/frontend/app/components/WaterBadge.tsx`.
- [X] T018 [P] [US3] Prevent redundant repeated checks per text (cache/throttle) to keep responses consistent in `/home/termenater/vibe/frontend/app/components/AnalyzePageClient.tsx`.

**Checkpoint**: Invalid/failed checks return clear feedback and UI remains responsive with neutral fallback.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Multi-story improvements and documentation.

- [ ] T019 [P] Add water detection contract reference and usage note to `/home/termenater/vibe/specs/001-water-detection/quickstart.md`.
- [ ] T020 Document endpoint and UI behavior in `/home/termenater/vibe/backend/README.md` and `/home/termenater/vibe/frontend/README.md` (or existing docs).
- [ ] T021 Run through quickstart validation flow (backend start, curl, frontend render) and note outcomes in `/home/termenater/vibe/specs/001-water-detection/quickstart.md`.

---

## Dependencies & Execution Order

- **Phase order**: Setup → Foundational → US1 (P1) → US2 (P2) → US3 (P3) → Polish.
- **Story dependencies**: US1 relies on the water API from US2 for live data (UI can be mocked during development but final acceptance needs US2). US3 builds on US2 for error/fallback handling and touches US1 UI states.

---

## Parallel Opportunities

- Setup: T001–T002 can run in parallel.
- Foundational: T003–T005 can run in parallel after Setup.
- US1: T006 (proxy), T007 (component), T008 (state), T009 (render) can mostly proceed in parallel once proxy target is known.
- US2: T010 (schemas), T011 (service), T012 (route), T013 (app include), T014 (proxy alignment) — schemas/service/route can be parallelized with clear contracts.
- US3: T015–T018 can run in parallel since validation, fallback, and client caching touch different areas.
- Polish: T019–T021 can run after stories stabilize.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup + Foundational.
2. Implement US1 UI with mocked/stubbed API responses if US2 not ready.
3. Integrate with live API once US2 is complete; validate UI label/confidence flow.

### Incremental Delivery

1. Setup → Foundational.
2. US1 (UI) + US2 (API) in tight loop; ship MVP once both green.
3. US3 adds robustness for validation/failures.
4. Polish docs/quickstart.

### Parallel Team Strategy

- Dev A: Backend config/service/schemas/routes (T003–T005, T010–T013).
- Dev B: Frontend proxy + components/state/render (T006–T009, T014).
- Dev C: Validation/fallback/caching and docs (T015–T018, T019–T021).
