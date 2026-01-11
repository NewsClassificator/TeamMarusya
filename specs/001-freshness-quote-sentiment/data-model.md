# Data Model: Article Freshness & Quote Sentiment

## Entities

- **ArticleAnalysisRequest**
  - Fields: `input_type` (enum url|text), `url` (optional string when input_type=url), `text` (optional string when input_type=text), `published_date` (optional string, ISO-like), `language` (string), `request_id` (optional string).
  - Rules: Reject empty `text` for text mode; reject missing `url` for url mode; accept `published_date` only if parseable and not in the future.
  - Relationships: Input to analysis pipeline.

- **FreshnessResult**
  - Fields: `status` (enum: today, yesterday, recent, stale, unknown), `age_days` (number|null), `reference_date` (string date used for comparison), `message` (string), `source_date` (string|null).
  - Rules: Status derived from calendar-day delta; unknown for missing/future/unparseable dates.
  - Relationships: Returned with sentiment results for the same article.

- **SentimentResult**
  - Fields: `main_text` ({text, sentiment_label, confidence}), `quotes` (ordered list of quote-level results), `errors` (optional array of warning strings).
  - Rules: Quote order matches appearance; mainText sentiment uses text with quotes replaced by `цитата`.
  - Relationships: Returned alongside `FreshnessResult`.

- **QuoteSentiment**
  - Fields: `quote_text` (string), `sentiment_label` (string), `confidence` (number), `position` (index/order).
  - Rules: Extracted from quoted segments; tolerate unmatched quotes by skipping malformed pairs.
  - Relationships: Member of `SentimentResult.quotes`.

- **AnalysisMeta**
  - Fields: `contract_version` (string), `analysis_version` (string/model id), `analyzed_at` (timestamp), `seed` (int).
  - Rules: Present on every response for determinism/version tracking.
  - Relationships: Top-level metadata accompanying results.
