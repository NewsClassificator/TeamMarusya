from typing import Dict, Optional

from src.api.schemas import ArticleContent


def normalize_article(raw: Dict[str, Optional[str]]) -> ArticleContent:
    """
    Convert raw article dict from the legacy parser into the ArticleContent schema.
    """
    return ArticleContent(
        title=raw.get("title"),
        author=raw.get("author"),
        published_at=raw.get("date"),
        content=raw.get("text") or "",
    )


