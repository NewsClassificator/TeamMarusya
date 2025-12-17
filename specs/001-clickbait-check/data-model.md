# Data Model: Clickbait Headline Check

## Entities

### Headline
- **Attributes**:
  - `text`: string, required, trimmed, length 5–200 chars.
  - `language` (optional): string, language code when provided.
- **Validation**: reject empty/whitespace; enforce length bounds; best-effort support for mixed languages with graceful fallback.

### ClickbaitEvaluation
- **Attributes**:
  - `is_clickbait`: boolean derived from model output.
  - `score`: float 0–1 (higher = more clickbait).
  - `label`: string summary for UI (e.g., "clickbait", "neutral/likely not clickbait").
  - `confidence_note` (optional): string when confidence is low or evaluation unavailable.
  - `detector_version`: string identifying model checkpoint or tag.
  - `contract_version`: string identifying response schema version.
  - `evaluated_at`: timestamp (optional) for recency/debugging.
- **Relationships**: references a single Headline instance.
- **Validation**: `score` within [0,1]; `label` consistent with `is_clickbait`; provide `confidence_note` when score near threshold.

## Derived/Operational Notes
- Determinism: fixed model path/version; set seed and preprocessing rules in service to keep outputs stable.
- Error handling: for invalid headline input, return structured errors instead of partial evaluation objects.
