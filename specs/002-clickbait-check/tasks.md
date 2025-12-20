---

description: "Task list for Clickbait Headline Check feature"
---

# Tasks: Clickbait Headline Check

**Input**: Design documents from `/specs/002-clickbait-check/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not explicitly requested; focus on implementation tasks. Add tests later if needed.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure environments and assets are ready.

- [ ] T001 Sync backend dependencies with uv using backend/uv.lock in backend/
- [X] T002 Install frontend dependencies from frontend/package.json
- [X] T003 Verify clickbait model assets in code/klikbait/my_awesome_model and note expected path in code/klikbait/README.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core plumbing required before user stories.

- [X] T004 Create clickbait config/constants module (model path, threshold, contract/detector versions) in backend/src/lib/clickbait_config.py
- [X] T005 Define clickbait request/response Pydantic schemas aligned to contract in backend/src/api/schemas_clickbait.py

**Checkpoint**: Backend config and schemas ready for API and UI work.

---

## Phase 3: User Story 1 - See clickbait label on headline (Priority: P1) 🎯 MVP

**Goal**: Readers see clickbait/neutral labels beside each headline.

**Independent Test**: Load page with sample headlines; verify each shows a label and confidence indicator without blocking page render.

### Implementation for User Story 1

- [X] T006 [US1] Add clickbait proxy route in frontend/app/api/clickbait/route.ts to POST to backend /clickbait/analyze
- [X] T007 [P] [US1] Implement clickbait badge component with label and confidence display in frontend/app/components/ClickbaitBadge.tsx
- [X] T008 [US1] Integrate clickbait fetch + badge rendering into headline view in frontend/app/components/AnalysisResult.tsx (call proxy per headline, show adjacency)
- [X] T009 [US1] Wire client-side state/async handling for badge loading/error states in frontend/app/components/AnalyzePageClient.tsx

**Checkpoint**: Page renders headlines with clickbait badges using API responses.

---

## Phase 4: User Story 2 - Request clickbait check via API (Priority: P2)

**Goal**: Internal consumers can POST a headline and receive deterministic clickbait results.

**Independent Test**: POST a headline to `/clickbait/analyze` and receive structured response with label, score, and versions.

### Implementation for User Story 2

- [X] T010 [US2] Implement clickbait detector service wrapping code/klikbait/predict.py in backend/src/services/clickbait_detector.py
- [X] T011 [US2] Add FastAPI router for POST /clickbait/analyze in backend/src/api/routes_clickbait.py using schemas/service
- [X] T012 [US2] Register clickbait router in backend/src/api/app.py and ensure contract version fields populated from config
- [X] T013 [US2] Ensure determinism/seed handling and consistent scoring in backend/src/lib/determinism.py or related config

**Checkpoint**: Clickbait API endpoint responds with structured results matching contract.

---

## Phase 5: User Story 3 - Graceful handling of invalid or failed checks (Priority: P3)

**Goal**: Invalid input or backend failures yield clear feedback without breaking the page.

**Independent Test**: Send empty/oversized headline or simulate detector failure; UI shows neutral/fallback message while page remains usable.

### Implementation for User Story 3

- [X] T014 [US3] Enforce headline validation rules (trim, length bounds, safe chars) with clear error codes in backend/src/api/schemas_clickbait.py
- [X] T015 [US3] Add timeout/failure handling and neutral fallback response shaping in backend/src/services/clickbait_detector.py
- [X] T016 [US3] Surface error/timeout fallback text and low-confidence notes in frontend/app/components/ClickbaitBadge.tsx and frontend/app/components/AnalyzePageClient.tsx
- [X] T017 [US3] Deduplicate repeated headline checks within a page session to keep labels consistent in frontend/app/components/AnalyzePageClient.tsx

**Checkpoint**: UI and API handle invalid inputs and failures gracefully with user-friendly messaging.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finalize docs, accessibility, and resilience across stories.

- [X] T018 [P] Update quickstart instructions with clickbait endpoint usage in specs/002-clickbait-check/quickstart.md
- [X] T019 [P] Add accessibility notes (aria labels/text) for badges in frontend/app/components/ClickbaitBadge.tsx
- [X] T020 Capture clickbait contract alignment and examples in specs/002-clickbait-check/contracts/clickbait-analyze.yaml (example payload/response)

---

## Dependencies & Execution Order

- Foundational (Phase 2) depends on Setup (Phase 1).  
- User Story dependencies: US1 badge rendering depends on US2 API availability; US3 builds on US1+US2 for error states.  
- Recommended order: Phase 1 → Phase 2 → US2 → US1 → US3 → Polish. If parallelizing, backend (US2) and frontend scaffolding (US1 T006–T007) can proceed in parallel once contracts are fixed.

## Parallel Execution Examples

- Run T010 (backend detector service) in parallel with T007 (frontend badge component) since they touch different stacks once schemas are defined.  
- Run T011–T012 (backend router + registration) in parallel with T008 (headline integration) after contract is finalized.  
- Run T018–T020 (docs/accessibility) in parallel after primary story tasks stabilize.

## Implementation Strategy (MVP First)

- MVP = US2 + minimal US1: ship POST `/clickbait/analyze` and render basic badge beside headlines using score/label; fallback can be minimal neutral text.  
- Incrementally add US3 resilience (validation, fallbacks, dedup) and polish (accessibility, docs).
