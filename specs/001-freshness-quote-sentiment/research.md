# Research: Article Freshness & Quote Sentiment

## Decisions

- **Decision**: Freshness statuses map to calendar-day deltas: 0 days → `today`; 1 day → `yesterday`; 2–7 days → `recent`; >7 days → `stale`; unparseable/missing/future → `unknown`.
  **Rationale**: Matches user expectation from spec examples and keeps thresholds simple and testable.
  **Alternatives considered**: (1) Hour-level granularity (rejected: unnecessary complexity); (2) Longer “recent” window (rejected: reduces clarity on outdated content).

- **Decision**: Compare publish date using naive date (no timezone offset) unless an explicit timezone is supplied; if timezone present, convert to local date before comparison.
  **Rationale**: Spec requires calendar-day comparison; keeps off-by-one minimal while supporting TZ input when provided.
  **Alternatives considered**: (1) Force UTC for all dates (rejected: could mislabel local publications); (2) Require timezone always (rejected: not available for many sources).

- **Decision**: Identify quotes using standard double/single quotation marks; non-matching pairs are treated as plain text and remain in main text.
  **Rationale**: Balances robustness and simplicity; avoids overfitting to edge punctuation.
  **Alternatives considered**: (1) Regex for nested quotes (rejected: brittle); (2) Removing unmatched quotes (rejected: might drop meaningful text).

- **Decision**: Replace each quote with the placeholder word `цитата` in the main-text body before main-text sentiment; run quote-level sentiment on each extracted quote independently and preserve original order.
  **Rationale**: Matches user request and keeps main-text sentiment focused on non-quote tone.
  **Alternatives considered**: (1) Remove quotes entirely (rejected: could distort sentence flow); (2) Keep quotes in main text (rejected: would skew combined sentiment).

- **Decision**: Sentiment outputs include label + confidence for main text and each quote, plus deterministic ordering and reference timestamp/model version.
  **Rationale**: Supports reproducibility and frontend labeling clarity.
  **Alternatives considered**: (1) Label-only (rejected: loses interpretability); (2) Aggregate sentiment only (rejected: spec requires per-quote detail).

- **Decision**: Graceful degradation: if date parse fails or sentiment model errors, return `unknown` freshness or neutral/unknown sentiment while still rendering available results.
  **Rationale**: Aligns with spec requirement for explicit “not available” states.
  **Alternatives considered**: (1) Hard error response (rejected: degrades UX); (2) Silent omission (rejected: confuses users).

- **Decision**: Versioning fields `contractVersion` and `analysisVersion` included in all responses; sentiment model hash/version recorded once per response.
  **Rationale**: Satisfies constitution for explicit interfaces and deterministic execution.
  **Alternatives considered**: (1) No versioning (rejected: breaks compatibility checks); (2) Version in headers only (rejected: harder for clients to consume).

## Open Questions Resolved

No outstanding NEEDS CLARIFICATION items; assumptions documented in the spec remain acceptable defaults.
