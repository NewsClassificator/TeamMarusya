---

description: "Task list template for feature implementation"
---

# Tasks: News Analysis MVP

**Input**: Design documents from `/specs/001-news-analysis-mvp/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Include contract/integration/unit checks as listed.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create backend skeleton directories `backend/src/{api,services,models,cli}` and `backend/tests/{unit,integration,contract}`
- [ ] T002 Initialize Python environment with uv and add dependencies (FastAPI, uvicorn, requests, beautifulsoup4 or newspaper3k, torch, transformers, pydantic) in `backend/pyproject.toml`
- [ ] T003 Initialize frontend dependencies in `frontend/` and set `NEXT_PUBLIC_API_BASE` handling in `.env.local.example`
- [ ] T004 Add shared constants for `contract_version`, `model_version`, and deterministic seed helper in `backend/src/lib/determinism.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [ ] T005 Create base FastAPI app with health endpoint in `backend/src/api/app.py`
- [ ] T006 Define Pydantic request/response schemas per contract in `backend/src/api/schemas.py`, enforcing exactly one of {url, text} or return 400
- [ ] T007 Implement HTTP server entry in `backend/src/api/main.py` wiring app and settings
- [ ] T008 Add fixture test config for CPU-only runs in `backend/tests/conftest.py`
- [ ] T009 Add basic contract test scaffolding for `/analyze` in `backend/tests/contract/test_analyze_contract.py` using local HTML/text fixtures (no live sites)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Request analysis (Priority: P1) 🎯 MVP

**Goal**: User submits URL or text and receives structured JSON with parsed fields and sentiment.

**Independent Test**: Post valid URL and text to `/analyze` and receive JSON containing article fields, sentiment, versions, and seed.

### Tests for User Story 1

- [ ] T010 [P] [US1] Write happy-path contract test for `/analyze` URL input in `backend/tests/contract/test_analyze_contract.py`
- [ ] T011 [P] [US1] Write happy-path contract test for `/analyze` text input in `backend/tests/contract/test_analyze_contract.py`
- [ ] T012 [P] [US1] Add integration test invoking real parsing+sentiment flow on sample text in `backend/tests/integration/test_analyze_flow.py`

### Implementation for User Story 1

- [ ] T013 [US1] Add article fetch wrapper to existing parser module with timeout config in `backend/src/services/fetcher.py`
- [ ] T014 [US1] Wrap existing HTML/article parser behind interface in `backend/src/services/parser_adapter.py`; add unit tests with local fixture HTML
- [ ] T015 [US1] Wrap existing fine-tuned RuBERT sentiment module in `backend/src/services/sentiment_adapter.py` with fixed seeds; add unit tests using fixture text
- [ ] T016 [US1] Implement orchestrator to combine fetch/parse/sentiment and assemble `AnalysisResponse` in `backend/src/services/analyzer.py`
- [ ] T017 [US1] Expose POST `/analyze` endpoint wiring schemas and orchestrator in `backend/src/api/routes.py` and include in app
- [ ] T018 [US1] Add CLI wrapper to trigger analysis locally from shell in `backend/src/cli/analyze.py`

**Checkpoint**: User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - View results (Priority: P2)

**Goal**: UI renders the analysis response clearly without modifying logic.

**Independent Test**: Trigger analysis and verify UI shows title, author, date, text, sentiment as returned.

### Implementation for User Story 2

- [ ] T019 [US2] Add API proxy route for POST `/analyze` in `frontend/app/api/analyze/route.ts` forwarding to backend URL
- [ ] T020 [US2] Build form component for URL/text input and submit handler in `frontend/app/components/AnalyzeForm.tsx`
- [ ] T021 [US2] Build results display component showing all returned fields verbatim in `frontend/app/components/AnalysisResult.tsx`
- [ ] T022 [US2] Wire page layout to render form and results flow in `frontend/app/page.tsx`

**Checkpoint**: User Story 2 should be fully functional and testable independently

---

## Phase 5: User Story 3 - Handle failures and determinism (Priority: P3)

**Goal**: Clear errors for bad input and consistent outputs across runs.

**Independent Test**: Submit unreachable URL and malformed payload to see clear errors; repeat valid input 5 times and observe identical sentiment outputs.

### Tests for User Story 3

- [ ] T023 [P] [US3] Add contract tests for validation errors (missing url/text) in `backend/tests/contract/test_analyze_contract.py`
- [ ] T024 [P] [US3] Add integration test asserting identical outputs for repeated input (seed/model_version) in `backend/tests/integration/test_determinism.py`

### Implementation for User Story 3

- [ ] T025 [US3] Implement error mapping for fetch timeouts/unreachable URLs with codes in `backend/src/services/fetcher.py`
- [ ] T026 [US3] Ensure deterministic seed assignment per request and echo in responses in `backend/src/services/analyzer.py`
- [ ] T027 [US3] Add frontend error state display for API errors in `frontend/app/components/AnalysisResult.tsx`
- [ ] T028 [US3] Add retry/duplicate submission guard on UI (disable submit during in-flight) in `frontend/app/components/AnalyzeForm.tsx`

**Checkpoint**: User Story 3 should be fully functional and testable independently

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T029 Document local run steps and determinism check in `specs/001-news-analysis-mvp/quickstart.md`
- [ ] T030 Add README snippets for backend and frontend entrypoints in `backend/README.md` and `frontend/README.md`
- [ ] T031 Manual verification: run identical request twice and capture responses for reference in `specs/001-news-analysis-mvp/artifacts/determinism-sample.json`

---

## Dependencies & Execution Order

- Story order: US1 → US2 → US3
- Frontend work depends on backend `/analyze` availability (US1 completion).

### Parallel Example: User Story 1

```bash
# Parallelizable tasks:
# - T010 (contract test URL) and T011 (contract test text) can be authored concurrently.
# - T013 (fetcher) and T015 (sentiment) can be built in parallel before orchestration.
```

### Implementation Strategy

1. Complete Setup + Foundational  
2. Deliver US1 end-to-end (backend + basic CLI)  
3. Add UI rendering (US2)  
4. Harden errors/determinism (US3)  
5. Polish docs and samples
