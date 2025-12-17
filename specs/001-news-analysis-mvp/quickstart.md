# Quickstart: News Analysis MVP

**Branch**: 001-news-analysis-mvp  
**Goal**: Run backend (FastAPI) and frontend (Next.js) locally on CPU-only Linux.

## Prerequisites
- Python 3.11, uv installed
- Node.js (matches Next.js app), npm or pnpm
- No GPU or cloud dependencies required

## Backend (Python)
```bash
cd backend
uv sync
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
- Endpoint: POST http://localhost:8000/analyze
- Contract: specs/001-news-analysis-mvp/contracts/openapi.yaml

Example request:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_type":"url","url":"https://example.com/news"}'
```

## Frontend (Next.js thin client)
```bash
cd frontend
npm install
npm run dev
```
- Opens http://localhost:3000
- Configure backend URL via env: `NEXT_PUBLIC_API_BASE=http://localhost:8000`

## Determinism Check
Run the same request twice and confirm identical `sentiment.score`, `sentiment.label`, `model_version`
and `seed` in responses.

## Troubleshooting
- Unreachable URL → backend returns 400 with error code; UI should display message verbatim.
- Long articles → ensure request still completes under the 5s target on CPU; trim content only if
  backend logic enforces limits.
