# Quickstart — Water Content Detection

**Feature**: /home/termenater/vibe/specs/001-water-detection/spec.md  
**Date**: 2025-12-22

## Prereqs
- Python 3.11 with project dependencies installed (includes FastAPI/uvicorn, joblib, pandas, pymorphy3, nltk stopwords).
- Existing model artifact for `WaterAnalyzer` available at the configured path (default `logreg_water_model.pkl`).
- Node/Next.js environment for frontend to consume the endpoint.

## Run backend locally
1. Activate project environment.
2. Start the API server (example):
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   Ensure the app mounts the `/api/water-detection` route that calls `WaterAnalyzer`.

## Call the endpoint
```bash
curl -X POST http://localhost:8000/api/water-detection \
  -H "Content-Type: application/json" \
  -d '{"text":"Ваш пример текста для проверки на воду...", "include_features": true}'
```
Expected response (shape):
```json
{
  "is_water": false,
  "label": "НЕ ВОДА",
  "confidence": 0.82,
  "features": {
    "readability_index": 65.1,
    "stopword_ratio": 0.28,
    "adj_ratio": 0.11,
    "adv_ratio": 0.04,
    "repetition_ratio": 0.03
  },
  "interpretations": {
    "readability": "нормально читается",
    "stopwords": "нормально"
  },
  "version": "water-detector-1.0"
}
```

## Frontend integration notes
- Trigger the POST from the existing frontend service layer and display label + confidence inline with the analyzed text.
- Show interpretations on demand (e.g., expandable details).
- On timeout or 4xx/5xx, display a neutral “status unavailable” message without blocking other content.

## Validation
- Confirm 95% of valid submissions render a label within 2s in local testing.
- Verify consistent responses for repeat submissions of the same text within a session.
- Add backend contract test for request validation and response schema shape; add frontend smoke test for rendering label/fallback.
