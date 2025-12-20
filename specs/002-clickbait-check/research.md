# Research: Clickbait Headline Check

## Decision 1: Use existing inference assets (`predict.py` + `my_awesome_model`) without retraining
- **Rationale**: The repository contains a packaged model and inference wrapper; no `train_clickbait.py` is present. Using shipped weights keeps delivery fast and aligns with “Python as Source of Truth” while meeting offline constraints.
- **Alternatives considered**:
  - Add/locate `train_clickbait.py` and retrain: rejected for scope creep and missing script.
  - Swap to a different pretrained model: rejected to avoid new dependencies and re-validation.

## Decision 2: Expose POST `/clickbait/analyze` as a dedicated endpoint
- **Rationale**: Keeps the existing `/analyze` contract stable while providing a clear clickbait-specific API for the frontend; supports future reuse by other clients.
- **Alternatives considered**:
  - Add fields to `/analyze`: rejected to avoid mixing sentiment and clickbait concerns and to minimize regression risk.
  - Create a separate service process: rejected to avoid extra deployment complexity; FastAPI app can host both.

## Decision 3: Response schema includes `is_clickbait`, `score`, `label`, `confidence_note`, and version fields
- **Rationale**: Provides deterministic, testable data with UX-friendly text and transparency on model/contract versions; aligns with spec success criteria.
- **Alternatives considered**:
  - Return raw model label only: rejected due to ambiguity and lack of confidence reporting.
  - Omit versioning: rejected because constitution requires explicit contracts and deterministic behavior.

## Decision 4: Frontend renders per-headline badge with loading/error fallbacks
- **Rationale**: Ensures non-blocking page render and accessible messaging; supports edge cases like timeouts or low confidence without degrading layout.
- **Alternatives considered**:
  - Block page until all results load: rejected to preserve responsiveness.
  - Hide errors silently: rejected because users need clarity when status is unavailable.
