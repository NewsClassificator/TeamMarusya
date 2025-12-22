# vibe Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-12-14

## Active Technologies
- Python 3.11; Next.js (App Router) + FastAPI, uvicorn, requests, BeautifulSoup/newspaper3k (parsing), torch + (001-news-analysis-mvp)
- None (in-memory, no persistence) (001-news-analysis-mvp)
- Python 3.11 (backend), TypeScript/Next.js (frontend) + FastAPI + Pydantic; Hugging Face transformers/torch for clickbait model; Next.js App Router for UI (002-clickbait-check)
- None (in-memory only) (002-clickbait-check)
- Python 3.11 for backend analysis; TypeScript/Next.js (App Router) for frontend + FastAPI/uvicorn HTTP layer, requests + BeautifulSoup/newspaper3k for parsing, Hugging Face transformers/torch for sentiment, Pydantic for request/response models (001-freshness-quote-sentiment)
- Python 3.11 (backend), TypeScript/Next.js App Router (frontend) + FastAPI/uvicorn for HTTP layer, WaterAnalyzer (joblib + pandas + pymorphy3 + nltk), requests/BS4/newspaper3k already present for other features (001-water-detection)

- (001-news-analysis-mvp)

## Project Structure

```text
backend/
frontend/
tests/
```

## Commands

# Add commands for 

## Code Style

: Follow standard conventions

## Recent Changes
- 001-water-detection: Added Python 3.11 (backend), TypeScript/Next.js App Router (frontend) + FastAPI/uvicorn for HTTP layer, WaterAnalyzer (joblib + pandas + pymorphy3 + nltk), requests/BS4/newspaper3k already present for other features
- 001-freshness-quote-sentiment: Added Python 3.11 for backend analysis; TypeScript/Next.js (App Router) for frontend + FastAPI/uvicorn HTTP layer, requests + BeautifulSoup/newspaper3k for parsing, Hugging Face transformers/torch for sentiment, Pydantic for request/response models
- 002-clickbait-check: Added Python 3.11 (backend), TypeScript/Next.js (frontend) + FastAPI + Pydantic; Hugging Face transformers/torch for clickbait model; Next.js App Router for UI


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
