import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status

from src.api.schemas import (
    AnalysisMeta,
    AnalyzeRequest,
    AnalyzeResponse,
    ArticleContent,
    FreshnessResult,
    SentimentResult,
    SentimentSummary,
    QuoteSentiment,
)
from src.lib.determinism import (
    CONTRACT_VERSION,
    MODEL_VERSION,
    create_determinism_context,
)
from .fetcher import FetchError, fetch_article
from .parser_adapter import normalize_article
from .sentiment_adapter import (
    analyze_sentiment_segments,
    get_model_version,
)


def _ensure_code_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    code_dir = repo_root / "code"
    code_dir_str = str(code_dir)
    if code_dir.is_dir() and code_dir_str not in sys.path:
        sys.path.insert(0, code_dir_str)


_ensure_code_on_path()

from components.freshness import assess_freshness  # type: ignore  # noqa: E402
from parser.quotes_test import (  # type: ignore  # noqa: E402
    find_quotes_and_authors,
    replace_quotes_with_placeholder,
)


def analyze_request(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Orchestrate fetching/parsing (for URLs) or using raw text, then sentiment analysis,
    freshness scoring, and assemble the AnalyzeResponse object.
    """
    ctx = create_determinism_context()
    errors: list[str] = []

    # Build article content
    article: ArticleContent
    if payload.input_type == "url":
        article = _article_from_url(payload.url)
    else:
        article = _article_from_text(payload.text, payload.published_date)

    if not article.content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_CONTENT", "message": "Article content is empty"},
        )

    # Freshness calculation
    freshness_raw = assess_freshness(article.published_at)
    if freshness_raw.status == "unknown":
        errors.append(freshness_raw.message)
    freshness = FreshnessResult(
        status=freshness_raw.status,
        age_days=freshness_raw.age_days,
        reference_date=freshness_raw.reference_date.isoformat(),
        message=freshness_raw.message,
        source_date=freshness_raw.source_date.isoformat() if freshness_raw.source_date else None,
    )

    # Quote extraction and replacement
    quotes_with_authors = find_quotes_and_authors(article.content)
    quotes = [q["quote"] for q in quotes_with_authors]
    if not quotes:
        errors.append("Цитаты не найдены в тексте")
    main_text = replace_quotes_with_placeholder(article.content)

    # Sentiment analysis
    try:
        sentiment_raw = analyze_sentiment_segments(main_text, quotes)
    except Exception as exc:  # pragma: no cover - defensive fallback
        errors.append("Не удалось выполнить анализ тональности")
        sentiment_raw = {
            "main_text": {
                "text": main_text,
                "sentiment_label": "neutral",
                "confidence": 0.0,
            },
            "quotes": [],
            "errors": [str(exc)],
        }

    sentiment = SentimentResult(
        main_text=SentimentSummary(**sentiment_raw["main_text"]),
        quotes=[
            QuoteSentiment(
                **quote,
                author=quotes_with_authors[idx]["authors"][0]
                if quotes_with_authors[idx]["authors"]
                else None,
            )
            for idx, quote in enumerate(sentiment_raw["quotes"])
        ],
        errors=sentiment_raw.get("errors", []),
    )

    errors.extend(sentiment.errors)

    return AnalyzeResponse(
        request_id=payload.request_id,
        article=article,
        freshness=freshness,
        sentiment=sentiment,
        meta=AnalysisMeta(
            contract_version=CONTRACT_VERSION,
            analysis_version=get_model_version(),
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            seed=ctx.seed,
        ),
        errors=errors,
    )


def _article_from_url(url: Optional[str]) -> ArticleContent:
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MISSING_URL", "message": "url must be provided for input_type='url'"},
        )

    try:
        raw = fetch_article(url=url)
    except FetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "FETCH_ERROR", "message": str(exc)},
        ) from exc

    if raw.get("error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "FETCH_FAILED", "message": raw["error"]},
        )

    return normalize_article(raw)


def _article_from_text(text: Optional[str], published_date: Optional[str]) -> ArticleContent:
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MISSING_TEXT", "message": "text must be provided for input_type='text'"},
        )

    return ArticleContent(
        title=None,
        author=None,
        published_at=published_date,
        content=text,
    )
