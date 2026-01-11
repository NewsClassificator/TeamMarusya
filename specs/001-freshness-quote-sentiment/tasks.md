# Tasks: Article Freshness & Quote Sentiment

**Input**: Design documents from `/specs/001-freshness-quote-sentiment/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Include targeted backend tests for freshness and quote handling to guard regressions.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure environments and dependencies support freshness + quote sentiment work.

- [X] T001 Verify backend dependencies cover date parsing, sentiment, and spaCy usage; update `backend/pyproject.toml` if any required libs are missing.
- [X] T002 [P] Validate frontend toolchain is ready for contract changes (typegen or manual typing) in `frontend/package.json` and `frontend/tsconfig.json`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core contract and schema groundwork required before user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 Update API request/response schemas to include freshness block and quote/main-text sentiment structures in `backend/src/api/schemas.py`.
- [X] T004 [P] Align analysis contract definitions with implementation by importing `specs/001-freshness-quote-sentiment/contracts/analysis-api.yaml` fields into route typing in `backend/src/api/routes.py`.
- [X] T005 [P] Add/update frontend response typing to match new contract (freshness + quote sentiments) in `frontend/app/components/AnalysisResult.tsx` or a shared types file under `frontend/app/`.

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 - See article freshness (Priority: P1) 🎯 MVP

**Goal**: Compute freshness from publish date vs today and surface a clear status on the site.

**Independent Test**: Given an article with a known publish date, the API returns freshness status (today/yesterday/recent/stale/unknown) and the UI renders it with fallback when missing.

### Implementation for User Story 1

- [X] T006 [US1] Implement calendar-day freshness classification (today/yesterday/recent/stale/unknown) with UTC-safe parsing in `code/components/freshness.py`.
- [X] T007 [P] [US1] Add backend unit tests for freshness buckets and edge cases (future, malformed, missing) in `backend/tests/unit/test_freshness.py`.
- [X] T008 [US1] Integrate freshness computation into analysis pipeline and response assembly in `backend/src/services/analyzer.py` (use article.published_at, include reference date/message).
- [X] T009 [US1] Render freshness label and explanatory text with a "date unavailable" fallback in `frontend/app/components/AnalysisResult.tsx`.

**Checkpoint**: User Story 1 should be fully functional and testable independently.

---

## Phase 4: User Story 2 - View separate sentiment for quotes (Priority: P2)

**Goal**: Extract quotes, replace them with `цитата` in the main text, and return separate sentiments for quotes and remaining text.

**Independent Test**: Given text with multiple quotes, the API returns per-quote sentiment (ordered) plus main-text sentiment (quotes replaced), and the UI shows both distinctly.

### Implementation for User Story 2

- [X] T010 [US2] Refine quote extraction and placeholder replacement to tolerate unmatched quotes in `code/parser/quotes_test.py`.
- [X] T011 [P] [US2] Add backend tests covering quote parsing, placeholder replacement, and ordering in `backend/tests/unit/test_quotes.py`.
- [X] T012 [US2] Extend sentiment adapter to analyze quotes individually and main text with placeholders in `backend/src/services/sentiment_adapter.py` (or new helper module under `backend/src/services/`).
- [X] T013 [US2] Populate quote-level and main-text sentiment fields in API responses in `backend/src/services/analyzer.py`, ensuring deterministic ordering and confidence values.
- [X] T014 [US2] Update UI to display quote sentiments (per quote) and main-text sentiment separately with clear labels in `frontend/app/components/AnalysisResult.tsx`.

**Checkpoint**: User Stories 1 and 2 should both work independently.

---

## Phase 5: User Story 3 - Clear results presentation (Priority: P3)

**Goal**: Ensure clarity and fallback states when data is missing or processing fails.

**Independent Test**: With missing publish date or no quotes, UI shows explicit "not available" while still showing available sentiment/freshness; backend returns errors array without failing the whole request.

### Implementation for User Story 3

- [X] T015 [US3] Add graceful error/warning collection (e.g., missing date, no quotes, sentiment failure) to the analysis response in `backend/src/services/analyzer.py`.
- [X] T016 [P] [US3] Render warning/fallback states for missing date or quotes, and avoid empty sections in `frontend/app/components/AnalysisResult.tsx`.

**Checkpoint**: All user stories independently functional with clear fallbacks.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanups and documentation.

- [X] T017 [P] Update quickstart example request/response to reflect freshness and quote sentiments in `specs/001-freshness-quote-sentiment/quickstart.md`.
- [ ] T018 Run end-to-end validation of `/analysis` flow (backend + frontend) using sample article text, documenting any open issues in `specs/001-freshness-quote-sentiment/tasks.md` notes section.

---

## Dependencies & Execution Order

- Setup (Phase 1) → Foundational (Phase 2) → User Story phases (P1 → P2 → P3) → Polish.
- User stories can proceed in parallel after Phase 2, but US2 builds on freshness schema already established in Phase 2; US3 depends on response fields from US1/US2.

### User Story Dependencies

- US1: no story dependencies (after foundational).
- US2: depends on contract/schema from foundational; otherwise parallel to US1.
- US3: depends on US1/US2 outputs to render fallbacks consistently.

### Parallel Opportunities

- Tasks marked [P] can run concurrently (e.g., T002 with T001; T004 with T003; tests T007/T011 alongside implementation once relevant code exists; frontend typing T005 can proceed once contract shape is agreed).
- Different user stories can be staffed in parallel after foundational, keeping file ownership separate (backend vs frontend).

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1–2.
2. Deliver US1 freshness computation and UI rendering.
3. Validate freshness labels and fallbacks end-to-end.

### Incremental Delivery

1. US1 → validate freshness.
2. US2 → validate quote/main-text sentiments.
3. US3 → validate fallbacks and warnings.

### Parallel Team Strategy

- Developer A: Freshness pipeline (T006–T008) and tests (T007).
- Developer B: Quote sentiment extraction + API wiring (T010–T013) and tests (T011).
- Developer C: Frontend rendering updates for freshness and sentiments (T009, T014, T016) and quickstart/polish (T017–T018).
