# Feature Specification: Clickbait Headline Check

**Feature Branch**: `001-clickbait-check`  
**Created**: 2025-12-17  
**Status**: Draft  
**Input**: User description: "Добавь фичу: проверка заголовков на кликбейт. Используй уже готовый код из code/klikbait, сделай HTTP‑эндпоинт и вывод на фронте рядом с заголовком."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See clickbait label on headline (Priority: P1)

A reader opens a page with news headlines and immediately sees whether each headline is clickbait next to the title.

**Why this priority**: Delivers primary value by informing readers at a glance.

**Independent Test**: Load a page with sample headlines and confirm labels show beside each title without using other new functionality.

**Acceptance Scenarios**:

1. **Given** headlines are loaded, **When** the page renders, **Then** each headline shows a clickbait/neutral label beside the text.
2. **Given** a headline detected as clickbait, **When** the page shows it, **Then** the label clearly indicates clickbait with a confidence value or short explanation.

---

### User Story 2 - Request clickbait check via API (Priority: P2)

An internal consumer submits a headline text to an API and receives a clickbait decision with confidence.

**Why this priority**: Enables frontend integration and future reuse.

**Independent Test**: Call the API with a headline string and verify structured response without needing the UI.

**Acceptance Scenarios**:

1. **Given** a valid headline string, **When** it is sent to the endpoint, **Then** the response includes a deterministic label and confidence score.

---

### User Story 3 - Graceful handling of invalid or failed checks (Priority: P3)

A user submits an empty or unsupported headline and receives a clear, non-blocking message while the rest of the page remains usable.

**Why this priority**: Prevents broken experiences and ensures resilience.

**Independent Test**: Submit invalid input or force a backend error and confirm the UI shows a fallback message while headlines remain visible.

**Acceptance Scenarios**:

1. **Given** an empty headline, **When** a check is attempted, **Then** the system rejects the request with a helpful message and no label is shown.
2. **Given** the detector is temporarily unavailable, **When** the page loads headlines, **Then** the UI shows a neutral “status unavailable” note beside affected titles without breaking layout.

### Edge Cases

- Headline text is empty, whitespace-only, or exceeds maximum length; system rejects and shows user-friendly feedback.
- Headline uses unsupported characters or mixed languages; system returns best-effort label or a clear “cannot evaluate” message.
- Duplicate headline requests in quick succession; responses remain consistent and do not degrade page performance.
- Detector returns low confidence; UI communicates uncertainty (e.g., “low confidence”) instead of a binary claim.
- Backend response is delayed beyond the target time; UI times out gracefully and surfaces fallback text without blocking other content.

## Requirements *(mandatory)*

### Python ↔ Frontend Contract *(mandatory)*

- **Interface Type**: HTTP API entrypoint at `/api/clickbait-check` callable locally without external services.
- **Request Schema**: JSON body with `headline` (string, required, trimmed, length limit such as 5–200 chars) and optional `language` (string, default auto-detect); reject empty or oversized input.
- **Response Schema**: JSON with `is_clickbait` (boolean), `score` (number 0–1, higher = more clickbait), `label` (string summary), `confidence_note` (string for low-confidence cases), `version` (string), and `errors` (array of strings when applicable); consistent field order for deterministic snapshots; standard HTTP errors for 4xx/5xx with machine-readable messages.
- **Versioning**: Response includes detector version and contract version token; clients treat unspecified versions as backward-compatible default.

### Functional Requirements

- **FR-001**: System MUST expose an authenticated endpoint that evaluates a single headline and returns a clickbait decision with confidence data.
- **FR-002**: System MUST validate headline input (non-empty after trim, length within agreed bounds, safe characters) and return a clear error if invalid.
- **FR-003**: System MUST surface the clickbait result adjacent to each headline in the UI, showing at minimum a binary label and confidence indicator without obscuring the headline text.
- **FR-004**: System MUST handle detector timeouts or failures by returning a neutral status and displaying a user-facing fallback message while logging the issue for follow-up.
- **FR-005**: System MUST keep response times user-friendly (target under 1 second for typical headlines) and avoid blocking page rendering while checks complete.
- **FR-006**: System MUST ensure repeated checks of the same headline within a session return consistent labels to avoid user confusion.
- **FR-007**: System MUST provide accessible labeling (readable text or aria-friendly indicator) so screen readers convey clickbait status.

### Key Entities *(include if feature involves data)*

- **Headline**: Title text presented to users; attributes include text content, optional detected language, and length metadata for validation.
- **ClickbaitEvaluation**: Result of analyzing a headline; attributes include binary label, score (0–1), confidence note, detector version, and timestamp of evaluation for freshness and troubleshooting.

## Assumptions

- Existing clickbait detection logic is available locally and does not require new external services or network access.
- Headlines are already supplied by the current news feed flow; no new persistence is needed beyond current session handling.
- Access control follows current application rules; the endpoint is reachable by the internal frontend but not exposed publicly without existing authentication.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 95% of headlines on a page display a clickbait status within 1 second of page load under normal network conditions.
- **SC-002**: 98% of valid API requests return a structured success response on first attempt without manual retries during a test window of at least 100 requests.
- **SC-003**: In a curated sample review, at least 90% of clickbait labels align with editorial expectations across primary supported languages.
- **SC-004**: Less than 2% of page views show a fallback “status unavailable” message for headlines during the monitored beta period.
- **SC-005**: Accessibility review confirms screen readers announce the clickbait status for all rendered headlines.
