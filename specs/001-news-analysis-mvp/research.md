# Research: News Analysis MVP

**Feature**: specs/001-news-analysis-mvp/spec.md  
**Date**: 2025-12-14  
**Context**: Backend owns parsing + RuBERT sentiment; frontend is thin; CPU-only local; deterministic
outputs.

## Interaction Model (HTTP vs CLI)
- **Decision**: Use HTTP POST `/analyze` via FastAPI.
- **Rationale**: Simple integration with Next.js fetch, supports structured validation and clear
  contract versioning, easy local run with uvicorn.
- **Alternatives considered**: CLI invocation (harder to orchestrate from browser); GraphQL (overkill
  for single action).

## Parsing Approach
- **Decision**: Use `requests` + `BeautifulSoup`/`newspaper3k` fallback for article extraction.
- **Rationale**: Lightweight, CPU-friendly, works offline; BS4 allows resilient HTML parsing when
  metadata missing.
- **Alternatives considered**: Headless browser (too heavy for MVP); external parsing APIs (violates
  no-cloud constraint).

## Sentiment Model & Determinism
- **Decision**: RuBERT via `transformers` + `torch`, CPU inference with fixed seeds and explicit
  model_version constant.
- **Rationale**: Matches requirement (Russian), reproducible with seeds, widely supported; CPU mode
  acceptable for MVP latency target (<5s).
- **Alternatives considered**: Lighter distilled models (faster but may reduce quality); cloud
  sentiment APIs (violate Python ownership + no-cloud).

## Error Handling & Validation
- **Decision**: Pydantic request validation; structured error codes/messages; unreachable URL/timeouts
  surfaced as errors without crashing UI.
- **Rationale**: Keeps frontend thin and predictable; matches constitution’s explicit interface and
  deterministic behavior.
- **Alternatives considered**: Frontend-side validation or retries (would duplicate logic).

## Data Model
- **Decision**: Domain entities `ArticleInput`, `ArticleContent`, `SentimentResult`,
  `AnalysisResponse` aligned with spec.
- **Rationale**: Matches spec requirements and keeps separation between parsing, sentiment, and
  response metadata.
- **Alternatives considered**: Collapsing into a single dict (harder to test and extend).

## Local Execution & Tooling
- **Decision**: Use uv for environment management; run uvicorn locally; pytest for tests.
- **Rationale**: Lightweight, no cloud; deterministic local reproducibility.
- **Alternatives considered**: Docker-based workflow (not needed for MVP, adds overhead).
