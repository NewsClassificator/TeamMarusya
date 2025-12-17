# Feature Specification: News Analysis MVP

**Feature Branch**: `001-news-analysis-mvp`  
**Created**: 2025-12-14  
**Status**: Draft  
**Input**: User description: "I want to build an MVP web application for news analysis. The system consists of: - A Python backend that is the single source of truth. - A Next.js frontend that acts only as a thin client. Backend responsibilities: - Accept a URL to a news article or raw text. - Parse the article and extract: title, author, publication date, full text. - Perform sentiment analysis using a fine-tuned Russian RuBERT model. - Return a structured JSON result with all extracted fields and sentiment scores. Frontend responsibilities: - Provide a simple UI where a user can paste a link or text. - Send the input to the Python backend. - Display the structured analysis result returned by the backend. - No ML logic or parsing logic is implemented in the frontend. Technical constraints: - All core logic must be implemented in Python. - Communication between frontend and backend must be explicit (HTTP API or CLI with JSON I/O). - The backend must be runnable locally on Linux (CPU-only, no CUDA). - The system must be deterministic: same input produces the same output. Project goals: - Clear separation of concerns between frontend and backend. - Simple, debuggable architecture suitable for an academic MVP. - Code should be structured to allow future extensions (additional analysis modules). Out of scope for this iteration: - Deployment to cloud. - User authentication. - Real-time processing or streaming."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Request analysis (Priority: P1)

As a reader, I paste a news URL or raw text and request analysis to get a structured summary with
sentiment.

**Why this priority**: It delivers the core value of automated news analysis for the MVP.

**Independent Test**: Submit a valid URL and a valid text snippet; verify a structured JSON result
with parsed fields and sentiment scores is returned without frontend logic.

**Acceptance Scenarios**:

1. **Given** a reachable article URL, **When** the user requests analysis, **Then** the system returns
   title, author (if present), publication date (if present), full text, and sentiment scores in JSON.
2. **Given** raw text input, **When** the user requests analysis, **Then** the system returns
   sentiment scores and echoes the provided text in the response payload.

---

### User Story 2 - View results (Priority: P2)

As a reader, I view the returned analysis in the UI so I can quickly see the parsed fields and
sentiment.

**Why this priority**: Users must be able to consume the backend output without ambiguity.

**Independent Test**: Trigger an analysis and verify the UI renders all returned fields clearly
without adding or altering logic.

**Acceptance Scenarios**:

1. **Given** a successful analysis response, **When** the frontend renders results, **Then** title,
   author, publication date, article text, and sentiment scores are displayed without modification.

---

### User Story 3 - Handle failures and determinism (Priority: P3)

As a reader, I receive clear errors for bad input and consistent results for repeat analyses so I can
trust the tool.

**Why this priority**: Reliability and debuggability are core project goals.

**Independent Test**: Submit an unreachable URL and malformed input to confirm error messaging; run
the same valid input multiple times to confirm identical outputs.

**Acceptance Scenarios**:

1. **Given** an unreachable URL, **When** analysis is requested, **Then** the system returns a clear
   error explaining the failure without crashing the UI.
2. **Given** the same valid input repeated, **When** analysis runs multiple times, **Then** the JSON
   response content is identical across runs.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- URL unreachable, returns non-200, or times out.
- Article missing author or publication date metadata.
- Extremely long articles or text beyond expected limits.
- Input contains unsupported language; sentiment still returned deterministically with a note.
- Duplicate submissions of the same input return identical sentiment scores.

## Requirements *(mandatory)*

### Python ↔ Frontend Contract *(mandatory)*

- **Interface Type**: HTTP API entrypoint `/analyze` callable locally without cloud dependencies.
- **Request Schema**: JSON with `input_type` (url|text), `url` (required when input_type=url),
  `text` (required when input_type=text), optional `language` (default ru), and optional
  `request_id`. Requests missing required fields are rejected with validation errors.
- **Response Schema**: JSON containing `title`, `author`, `published_at` (ISO 8601 or null),
  `content`, `sentiment` (label and score), `request_id`, `model_version`, `contract_version`, and
  `seed`. Error responses include machine-readable codes and human-readable messages.
- **Versioning**: Contract version and model version are returned in every response; seeds are fixed
  per request to guarantee deterministic outputs.

### Functional Requirements

- **FR-001**: System MUST accept either a news URL or raw text as input from the UI and forward it to
  the backend without alteration.
- **FR-002**: System MUST retrieve article content for URL inputs and extract title, author, publish
  date (if present), and full text.
- **FR-003**: System MUST perform sentiment analysis on provided text using a fixed model and seed to
  ensure identical outputs for identical inputs.
- **FR-004**: System MUST return a structured JSON payload containing parsed fields, sentiment label,
  sentiment score, and metadata (model version, contract version, seed, request identifier).
- **FR-005**: System MUST handle invalid inputs (missing URL/text, unreachable URL, empty text) with
  clear error responses without crashing the frontend.
- **FR-006**: System MUST log request/response metadata sufficient for reproducibility (request_id,
  contract_version, model_version, seed) without storing user content beyond processing.

### Key Entities *(include if feature involves data)*

- **ArticleInput**: User-provided URL or raw text plus optional language and request_id.
- **ArticleContent**: Parsed article fields: title, author, publication date, full text.
- **SentimentResult**: Sentiment label and score derived from RuBERT for the provided content.
- **AnalysisResponse**: Combined payload of ArticleContent, SentimentResult, and reproducibility
  metadata (model_version, contract_version, seed, request_id).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive a structured analysis response (title, author/date if available, text,
  sentiment) within 5 seconds for typical news articles on a local CPU-only machine.
- **SC-002**: 100% of repeated analyses on identical input return identical sentiment labels and
  scores across at least 5 consecutive runs.
- **SC-003**: For reachable, supported news URLs, 95% return non-empty title and body text on first
  attempt; missing metadata is explicitly marked as null.
- **SC-004**: 100% of invalid inputs (missing fields, unreachable URLs, empty text) produce clear
  error responses without frontend crashes or silent failures.
