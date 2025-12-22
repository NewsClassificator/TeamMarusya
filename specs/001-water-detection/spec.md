# Feature Specification: Water Content Detection

**Feature Branch**: `001-water-detection`  
**Created**: 2025-12-22  
**Status**: Draft  
**Input**: User description: "новая фича, а именно определение воды в тексте. Находится в папке code/water, water_analyzer.py - основной файл. Нужно интегрировать эту фичу в сайт"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See water label on text (Priority: P1)

An editor or reader submits an article or paragraph and immediately sees whether it is “water” or “not water,” with a concise confidence indicator.

**Why this priority**: Delivers the core value of the feature and enables quick judgment of text quality.

**Independent Test**: Submit a sample text via the UI and confirm a binary water/not-water label and confidence appear without relying on any other new feature.

**Acceptance Scenarios**:

1. **Given** a valid text sample is provided, **When** the check is triggered, **Then** the page shows a water/not-water label with a confidence value near the text.
2. **Given** the label is displayed, **When** the user requests details, **Then** the UI reveals a brief explanation of the factors that led to the decision.

---

### User Story 2 - Request water check via API (Priority: P2)

An internal consumer sends text to a dedicated endpoint and receives a structured decision plus feature signals for further use in the UI.

**Why this priority**: Enables frontend integration and reuse by other internal clients without coupling to UI behavior.

**Independent Test**: Call the endpoint with a text string and verify a deterministic response containing decision, confidence, and feature breakdown without involving the page.

**Acceptance Scenarios**:

1. **Given** a valid request payload, **When** it is sent to the endpoint, **Then** the response includes a binary decision, confidence score, and feature metrics in a consistent format.

---

### User Story 3 - Graceful handling of invalid or failed checks (Priority: P3)

An editor submits empty, too-short, or unsupported text and receives a clear, non-blocking message while the rest of the page remains usable.

**Why this priority**: Prevents broken experiences and keeps the page usable during validation errors or backend issues.

**Independent Test**: Send invalid input or simulate a backend failure and confirm the UI shows guidance or fallback status without blocking other interactions.

**Acceptance Scenarios**:

1. **Given** an empty or too-short text, **When** a check is attempted, **Then** the system rejects the request with a helpful message and no label is shown.
2. **Given** the detector is temporarily unavailable, **When** the page requests a check, **Then** the UI shows a neutral “status unavailable” note without breaking layout.

---

### Edge Cases

- Text is empty, whitespace-only, or below the minimum length; request is rejected with clear feedback.
- Text exceeds maximum allowed length; request is trimmed or rejected with guidance and does not degrade page performance.
- Text includes mixed languages or unsupported characters; response returns best-effort analysis or a clear “cannot evaluate” message.
- Rapid repeated requests for the same text; responses remain consistent and do not overwhelm the page.
- Detector returns low confidence; UI communicates uncertainty instead of a binary claim.
- Backend response exceeds the target time; UI times out gracefully and surfaces fallback text without blocking other content.

## Requirements *(mandatory)*

### Python ↔ Frontend Contract *(mandatory)*

- **Interface Type**: HTTP API entrypoint at `/api/water-detection`, callable locally without external services.
- **Request Schema**: JSON body with `text` (string, required, trimmed, length guard such as 20–10,000 characters), optional `include_features` (boolean, default true), and optional `language` hint (string, for messaging only); reject empty, oversized, or unsupported payloads with structured errors.
- **Response Schema**: JSON with `is_water` (boolean), `label` (string summary such as “ВОДА”/“НЕ ВОДА”), `confidence` (number 0–1), `features` (object with readability index, stopword ratio, adjective ratio, adverb ratio, repetition ratio), optional `interpretations` (object with short human-readable hints), `version` (string), and `errors` (array of strings when applicable); consistent field ordering for deterministic snapshots; standard HTTP status codes for validation or server errors.
- **Versioning**: Responses include detector version and contract version token; clients treat unspecified versions as the default and remain backward-compatible.

### Functional Requirements

- **FR-001**: System MUST expose an endpoint that evaluates submitted text and returns a binary water/not-water decision with confidence.
- **FR-002**: System MUST validate text input (non-empty after trim, within agreed length bounds, safe characters) and return a clear error if invalid.
- **FR-003**: System MUST provide an option to return feature signals (readability, stopword ratio, adjective ratio, adverb ratio, repetition ratio) to support transparency in the UI.
- **FR-004**: System MUST surface the water detection result in the UI beside the analyzed text, showing at minimum a binary label and confidence without obscuring the text itself.
- **FR-005**: System MUST present brief, user-friendly interpretations of the feature signals when requested, so users understand why text is marked as water.
- **FR-006**: System MUST handle detector timeouts or failures by returning a neutral status and displaying a user-facing fallback message while keeping the page usable.
- **FR-007**: System MUST ensure repeated checks of the same text within a session return consistent labels to avoid confusion.
- **FR-008**: System MUST support evaluating multiple text samples in sequence without noticeably degrading page responsiveness.

### Key Entities *(include if feature involves data)*

- **TextSample**: A user-provided snippet or article body; attributes include raw text, detected/declared language hint, length metadata for validation, and submission timestamp for session context.
- **WaterEvaluation**: Result of analyzing a text; attributes include binary label, confidence score, feature metrics (readability, stopwords, adjectives, adverbs, repetition), optional interpretations, detector version, and evaluation timestamp.

## Assumptions

- Detection model and feature extraction are available locally and do not require new external services or storage.
- Primary language for evaluation is Russian; non-Russian text is handled on a best-effort basis with clear messaging if confidence is low.
- Existing site authentication and rate-limiting rules continue to apply; no new roles or permissions are introduced for this feature.
- UI will display results inline with the analyzed text rather than redirecting to a separate page.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 95% of valid text submissions display a water/not-water label with confidence in the UI within 2 seconds under normal conditions.
- **SC-002**: 98% of valid API requests return a structured success response on the first attempt during a sample of at least 100 requests.
- **SC-003**: In an editorial review of at least 50 diverse samples, 85% or more of labels align with expected judgments of wateriness.
- **SC-004**: Less than 3% of user interactions surface a fallback “status unavailable” message during the monitored beta period.
- **SC-005**: At least 90% of responses that include feature signals also display human-readable interpretations without missing values.
