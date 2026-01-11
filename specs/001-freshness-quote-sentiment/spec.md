# Feature Specification: Article Freshness & Quote Sentiment

**Feature Branch**: `001-freshness-quote-sentiment`  
**Created**: 2025-12-22  
**Status**: Draft  
**Input**: User description: "появилось 2 новые фичи их нужно внедрить в сайт: code/components/freshness.py нужно чтобы дату с сайта сравнивали с сегодняшней и выводило её свежесть. 2. code/parser/quotes_test.py нужно чтобы цитаты заменялись на цитата и прогонялся в нейронку с сентиментом уже не весь текст, а цитата отдельно и текст (без цитаты) отдельно. и чтобы результаты выводились пользователю на сайте"

## Assumptions

- Article publish dates use the site’s timezone; when timezone is absent, treat the date as local to the server and compare to the current calendar date.
- Quotes are text enclosed in standard quotation marks; malformed or unmatched quotes remain in the main text.
- When freshness or sentiment cannot be determined, the UI shows a clear "not available" state rather than leaving blanks.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See article freshness (Priority: P1)

A reader wants to know how recent the article is, so they can trust its relevance.

**Why this priority**: Recency directly affects user trust and relevance, making it critical for content evaluation.

**Independent Test**: Load an article with a known publish date and verify the page displays a human-readable freshness label derived from today’s date.

**Acceptance Scenarios**:

1. **Given** an article with a valid publish date, **When** the page loads, **Then** the freshness label shows an age such as "today", "yesterday", or "X days ago".
2. **Given** an article older than the recency threshold (e.g., older than a week), **When** the page loads, **Then** the freshness label clearly indicates it is older/outdated.

---

### User Story 2 - View separate sentiment for quotes (Priority: P2)

A reader wants to see the tone of quoted statements separately from the rest of the article to understand differing viewpoints.

**Why this priority**: Separating sentiment for quotes prevents quotes from skewing the overall tone and improves content clarity.

**Independent Test**: Submit article text containing at least one quote and verify the UI shows sentiment for each quote and for the remaining text independently.

**Acceptance Scenarios**:

1. **Given** article text with multiple quotes, **When** the page loads, **Then** each quote shows its own sentiment result, ordered as they appear.
2. **Given** the same article, **When** the page loads, **Then** the non-quote text shows its own sentiment result that excludes the quotes.

---

### User Story 3 - Clear results presentation (Priority: P3)

A reader wants to see freshness and sentiment results without confusion, even when data is missing.

**Why this priority**: Clear labeling reduces misinterpretation and improves trust, even in edge cases.

**Independent Test**: Load an article missing a publish date or quotes and verify the UI shows explicit "not available" states while still showing any available sentiment or freshness data.

**Acceptance Scenarios**:

1. **Given** an article missing a publish date, **When** the page loads, **Then** the freshness area shows an explicit "date unavailable" state.
2. **Given** article text with no quotes, **When** the page loads, **Then** the sentiment area shows only the main text sentiment with no empty or broken quote slots.

---

### Edge Cases

- Publish date missing, malformed, or in the future; freshness should fall back to "date unavailable" without blocking other results.
- Publish date near midnight or in a different timezone causing off-by-one-day differences; freshness uses calendar-day comparison to current date.
- Article contains no quotes, only quotes, or deeply nested/mismatched quotation marks; sentiment still returns consistent structures.
- Excessively long quotes or very short quotes; sentiment results remain ordered and labeled without truncation errors.
- Sentiment analysis fails or times out; UI shows a neutral/unknown state instead of partial or misleading values.

## Requirements *(mandatory)*

### Python ↔ Frontend Contract *(mandatory)*

- **Interface Type**: HTTP API entrypoint exposed locally (e.g., POST `/analysis`), callable without external cloud dependencies.
- **Request Schema**: JSON payload including `pageUrl` (string, optional), `publishedDate` (string, ISO-like date, optional), and `articleText` (string, required). Reject empty `articleText`; accept `publishedDate` only if it parses to a calendar date not more than one year in the future.
- **Response Schema**: JSON with `freshness` object `{status: "today"|"yesterday"|"recent"|"stale"|"unknown", ageDays: number|null, referenceDate: string, message: string}` and `sentiment` object containing `quotes` array of `{quoteText: string, sentimentLabel: string, confidence: number}` ordered by appearance, plus `mainText` object `{text: string, sentimentLabel: string, confidence: number}`. Include `errors` array for recoverable issues (e.g., missing date, no quotes found) while still returning available results.
- **Versioning**: Include `contractVersion` and `analysisVersion` fields in responses so UI and backend can ensure compatibility and trace which sentiment model configuration produced the results.

### Functional Requirements

- **FR-001**: System MUST calculate article freshness by comparing the provided publish date to the current calendar date and assign a user-friendly status (today, yesterday, recent, stale, or unknown).
- **FR-002**: System MUST display the freshness status and explanatory text in the UI alongside the analyzed article.
- **FR-003**: System MUST identify quoted segments in the article text and replace each with the placeholder word "цитата" in the non-quote text used for main-text sentiment.
- **FR-004**: System MUST compute sentiment for each quote independently and return quote-level results in the order they appear.
- **FR-005**: System MUST compute sentiment for the remaining text (with quotes removed/replaced) separately from quotes and return a distinct result.
- **FR-006**: System MUST present both quote-level and main-text sentiment results to users, clearly labeling which text each result refers to.
- **FR-007**: System MUST handle inputs with no quotes, only quotes, or malformed quotes without errors, returning appropriate empty or unknown sentiment states.
- **FR-008**: System MUST handle missing, future, or unparseable publish dates gracefully by returning an "unknown" freshness status and continuing sentiment analysis.
- **FR-009**: System MUST preserve deterministic ordering of sentiment outputs and include reference dates/timestamps so results can be correlated to the analysis moment.

### Key Entities *(include if feature involves data)*

- **ArticleAnalysisRequest**: Contains the source URL (optional), publish date (optional), and full article text to be analyzed for freshness and sentiment.
- **FreshnessResult**: Represents the computed age in days (or null), freshness status label, reference date used for comparison, and any explanatory message.
- **SentimentResult**: Represents sentiment for main text (with quotes replaced) and an ordered list of quote-level sentiments, each with text, label, confidence, and position.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For articles with a valid publish date, 95% of page loads display a freshness label derived from today’s date within one second of rendering the analysis results.
- **SC-002**: For articles containing at least one quote, 90% of page loads display separate sentiment outputs for each quote and for the main text on a single view without extra user actions.
- **SC-003**: For articles without quotes or without a publish date, 100% of page loads display explicit "not available" states while still showing any available sentiment or freshness data.
- **SC-004**: User feedback sessions report that at least 85% of participants can correctly explain whether the article is current and how the quote sentiment differs from the main text after viewing the results.
