# Data Model — Water Content Detection

**Feature**: /home/termenater/vibe/specs/001-water-detection/spec.md  
**Date**: 2025-12-22

## Entities

### TextSample
- **Description**: User-submitted text body for evaluation.
- **Fields**:
  - `id` (ephemeral request id; generated per request/session)
  - `text` (string; required; trimmed; length 20–10,000 chars)
  - `language_hint` (string; optional; informational only)
  - `submitted_at` (timestamp; request time for auditing/latency measurement)
- **Validation Rules**:
  - Must be non-empty after trimming whitespace.
  - Length must be within 20–10,000 characters; oversized payloads rejected.
  - Reject control characters outside standard text range; preserve Russian characters.

### WaterEvaluation
- **Description**: Result of analyzing a TextSample with the WaterAnalyzer.
- **Fields**:
  - `is_water` (boolean; primary label)
  - `label` (string; e.g., “ВОДА” / “НЕ ВОДА”)
  - `confidence` (float 0–1; probability of predicted class)
  - `features` (object):
    - `readability_index` (float)
    - `stopword_ratio` (float 0–1)
    - `adj_ratio` (float 0–1)
    - `adv_ratio` (float 0–1)
    - `repetition_ratio` (float 0–1)
  - `interpretations` (object; optional, short human-readable notes per feature)
  - `version` (string; detector/contract version token)
  - `evaluated_at` (timestamp)
- **Validation Rules**:
  - Confidence must match returned probabilities from analyzer.
  - Feature fields must be present when `include_features=true`; may be omitted otherwise.
  - Interpretations included only when available and mapped to feature fields.

## Relationships
- A **TextSample** has one **WaterEvaluation** result per submission.
- Repeated submissions of identical text may produce the same **WaterEvaluation** within a session; consistency required.
