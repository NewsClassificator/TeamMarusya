# News Analysis Backend

FastAPI backend for the News Analysis MVP feature.

## Local development

- Install dependencies with `uv sync` in the `backend/` directory.
- Run the server with:

```bash
uv run python -m src.api.main
```

## Water detection assets

- Water model artifact default path: `code/water/logreg_water_model.pkl` (relative to repo root).
- Water analyzer module: `code/water/water_analyzer.py` (loads the model and computes features).
- Ensure these files are present locally before starting the API; override paths via `src/lib/water_config.py`.
