# Quickstart: Article Freshness & Quote Sentiment

1) **Backend setup**
- Install dependencies in the Python environment (uv/venv): ensure FastAPI, uvicorn, torch, transformers, requests, BeautifulSoup/newspaper3k, pytest, pydantic are available.
- Confirm the analysis service exposes POST `/analysis` per `contracts/analysis-api.yaml`.

2) **Run locally**
- Start backend: `uvicorn backend.src.api:app --reload` (or project entrypoint) with model weights available locally.
- Start frontend: `npm install` (if needed) then `npm run dev` from `frontend/`.

3) **Manual test**
- Send a request:  
  ```bash
  curl -X POST http://localhost:8000/analysis \
    -H "Content-Type: application/json" \
    -d '{"input_type":"text","text":"\"Hello\" said John. Today is news.","published_date":"2025-12-21"}'
  ```
- Expect freshness status (today/yesterday/recent/stale/unknown), per-quote sentiment, and main-text sentiment with quotes replaced by `цитата`.

4) **Determinism checks**
- Verify response contains `meta.contract_version`, `meta.analysis_version`, and `meta.analyzed_at`.
- Re-run the same request and confirm identical labels/confidences.

5) **UI verification**
- Load the article page via Next.js UI; confirm freshness label and both quote-level and main-text sentiment render with clear “not available” fallbacks when data is missing.
