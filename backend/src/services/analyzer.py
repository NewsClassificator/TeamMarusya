from typing import Optional

from fastapi import HTTPException, status

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ArticleContent,
    SentimentResult,
)
from src.lib.determinism import create_determinism_context
from .fetcher import FetchError, fetch_article
from .parser_adapter import normalize_article
from .sentiment_adapter import (
    analyze_sentiment,
    get_model_version,
    map_label_to_contract,
)


def analyze_request(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Orchestrate fetching/parsing (for URLs) or using raw text, then sentiment analysis,
    and assemble the AnalyzeResponse object.
    """
    ctx = create_determinism_context()

    # Build article content
    article: ArticleContent
    if payload.input_type == "url":
        article = _article_from_url(payload.url)
    else:
        article = _article_from_text(payload.text)

    if not article.content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_CONTENT", "message": "Article content is empty"},
        )

    # Sentiment analysis
    sentiment_raw = analyze_sentiment(article.content)
    contract_label = map_label_to_contract(sentiment_raw.get("predicted_label", "NEUTRAL"))
    score = float(sentiment_raw.get("confidence", 0.0))

    sentiment = SentimentResult(label=contract_label, score=score)

    return AnalyzeResponse(
        request_id=payload.request_id,
        contract_version=ctx.contract_version,
        model_version=get_model_version(),
        seed=ctx.seed,
        article=article,
        sentiment=sentiment,
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


def _article_from_text(text: Optional[str]) -> ArticleContent:
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MISSING_TEXT", "message": "text must be provided for input_type='text'"},
        )

    return ArticleContent(
        title=None,
        author=None,
        published_at=None,
        content=text,
    )


