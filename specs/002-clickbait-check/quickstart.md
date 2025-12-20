# Quickstart: Clickbait Headline Check

## Backend (FastAPI)

1) Install deps  
```bash
cd /home/termenater/vibe/backend
uv sync
```

2) Ensure model assets exist  
- `code/klikbait/my_awesome_model` with config/tokenizer files.  
- Inference wrapper: `code/klikbait/predict.py`.

3) Run API  
```bash
cd /home/termenater/vibe/backend
uv run python -m src.api.main
```

4) Test endpoint  
```bash
curl -X POST http://localhost:8000/clickbait/analyze \
  -H "Content-Type: application/json" \
  -d '{"headline":"Шок! Ученые раскрыли тайну","language":"ru"}'
```

Expected: JSON with `is_clickbait`, `score`, `label`, `contract_version`, `detector_version`.

## Frontend (Next.js)

1) Install deps (if not already)  
```bash
cd /home/termenater/vibe/frontend
npm install
```

2) Start dev server  
```bash
npm run dev
```

3) Behavior  
- Headlines render immediately; clickbait badges load asynchronously from POST `/clickbait/analyze`.  
- Errors/timeouts show a neutral “status unavailable” message; low confidence shows a note.
- Frontend proxy endpoint: `/api/clickbait` forwards to the backend and honors `NEXT_PUBLIC_API_BASE` if set.

## Determinism
- Use the bundled model version and fixed preprocessing; avoid changing model files without updating `detector_version`.
- Keep the contract version in sync with `contracts/clickbait-analyze.yaml`.
