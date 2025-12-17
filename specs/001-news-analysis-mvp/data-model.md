# Data Model: News Analysis MVP

**Feature**: specs/001-news-analysis-mvp/spec.md  
**Date**: 2025-12-14

## Entities

### ArticleInput
- `input_type` (enum: url|text) — required
- `url` (string, required when input_type=url, must be valid URL)
- `text` (string, required when input_type=text, non-empty)
- `language` (string, default "ru")
- `request_id` (string, optional for tracing)

### ArticleContent
- `title` (string, nullable when missing)
- `author` (string, nullable when missing)
- `published_at` (ISO 8601 string, nullable)
- `content` (string, required full text)

### SentimentResult
- `label` (string, e.g., positive/neutral/negative)
- `score` (float 0-1)

### AnalysisResponse
- `request_id` (string)
- `contract_version` (string)
- `model_version` (string)
- `seed` (int)
- `article` (ArticleContent)
- `sentiment` (SentimentResult)
- `errors` (optional array of error objects when applicable)

### Error
- `code` (string, machine-readable)
- `message` (string, human-readable)
- `details` (optional object)

## Relationships & Constraints
- `ArticleInput` drives parsing (URL) or directly feeds sentiment (text).
- `AnalysisResponse.article` is required on success; absent on fatal parsing failure.
- Seeds and versions are echoed for determinism in every successful response.
- Validation rejects requests missing required fields or with empty text.
