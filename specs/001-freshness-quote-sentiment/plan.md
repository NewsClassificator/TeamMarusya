# Implementation Plan: Article Freshness & Quote Sentiment

**Branch**: `001-freshness-quote-sentiment` | **Date**: 2025-12-22 | **Spec**: specs/001-freshness-quote-sentiment/spec.md  
**Input**: Feature specification from `/specs/001-freshness-quote-sentiment/spec.md`

## Summary

Add freshness labeling that compares article publish dates to the current day and surfaces clear statuses (today/yesterday/recent/stale/unknown) on the site, and separate sentiment analysis for quotes versus main text by replacing quotes with the placeholder "цитата" before running main-text sentiment. Results (freshness, per-quote sentiment, main-text sentiment) will be returned from Python via an explicit contract and rendered in the frontend with graceful fallbacks when data is missing.

## Technical Context

**Language/Version**: Python 3.11 for backend analysis; TypeScript/Next.js (App Router) for frontend  
**Primary Dependencies**: FastAPI/uvicorn HTTP layer, requests + BeautifulSoup/newspaper3k for parsing, Hugging Face transformers/torch for sentiment, Pydantic for request/response models  
**Storage**: None (in-memory only)  
**Testing**: pytest for backend; frontend testing not in scope for this change unless added via existing test harness  
**Target Platform**: Linux server / local dev; served via Next.js frontend consuming local Python service  
**Project Type**: Web (backend + frontend)  
**Performance Goals**: Freshness and sentiment responses available to frontend within ~1s for typical article lengths; deterministic outputs for identical inputs  
**Constraints**: Offline/local execution without new cloud dependencies; deterministic model/config versions; frontend remains display/orchestration only  
**Scale/Scope**: Single-site analysis flow; no multi-tenant or persistence requirements

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Python as Source of Truth**: All parsing, freshness calculation, and sentiment processing stay in Python, callable via HTTP/CLI without frontend logic duplication. ✅
- **Explicit Interface**: Request/response schemas will be documented and versioned before frontend integration. ✅
- **Single Responsibility**: Freshness logic and sentiment processing remain in their respective Python modules; frontend only renders. ✅
- **Frontend as Thin Client**: No model or parsing logic in Next.js; UI consumes Python responses and displays labels. ✅
- **Deterministic Execution**: Model/version identifiers and preprocessing steps (quote replacement, date handling) will be fixed and documented; no cloud variance. ✅

## Project Structure

### Documentation (this feature)

```text
specs/001-freshness-quote-sentiment/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (not created in this command)
```

### Source Code (repository root)

```text
backend/
├── src/                 # Python logic: parsing, sentiment, freshness
└── tests/               # pytest suites (unit/integration as available)

frontend/
├── src/                 # Next.js App Router UI
└── tests/               # frontend tests (if present)

tests/                   # repo-level tests (existing)
```

**Structure Decision**: Use existing backend/ and frontend/ separation; all analysis logic remains under backend (Python), UI changes under frontend (Next.js).

## Complexity Tracking

No constitution violations identified; table empty.

## Phase 0: Outline & Research

- Unknowns identified: None blocking; assumptions from spec stand (date timezone fallback, placeholder text `цитата`, graceful degradation).
- Research tasks executed: validated freshness thresholds, quote handling rules, deterministic metadata, fallback behaviors.
- Output: specs/001-freshness-quote-sentiment/research.md (decisions on freshness buckets, timezone handling, quote parsing tolerance, sentiment versioning, degradation strategy).

## Phase 1: Design & Contracts

- Data model captured in specs/001-freshness-quote-sentiment/data-model.md.
- API contract defined at specs/001-freshness-quote-sentiment/contracts/analysis-api.yaml (POST `/analysis` with freshness + sentiment + metadata).
- Quickstart guidance documented at specs/001-freshness-quote-sentiment/quickstart.md.
- Agent context: to be updated via `.specify/scripts/bash/update-agent-context.sh codex` after code alignment (no new tech beyond constitution stack).
- Constitution re-check: passes with explicit contract, Python-owned logic, and deterministic requirements documented.

## Phase 2: Implementation Planning (up next)

- Backend tasks (Python): implement freshness calculation (calendar-day buckets), quote extraction/placeholder replacement, per-quote and main-text sentiment, deterministic metadata, error handling.
- Frontend tasks (Next.js): render freshness label with fallbacks; display per-quote and main-text sentiment; surface warnings/errors.
- Testing: add pytest coverage for freshness edge cases, quote parsing variations, and response contract; add frontend rendering check if harness exists.
