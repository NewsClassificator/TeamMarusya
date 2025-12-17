from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ClickbaitAnalyzeRequest(BaseModel):
    headline: str = Field(..., min_length=5, max_length=200, description="Headline text to evaluate (trimmed).")
    language: Optional[str] = Field(default=None, description="Optional language code; defaults to auto/best-effort.")

    @model_validator(mode="after")
    def validate_headline(self) -> "ClickbaitAnalyzeRequest":
        text = (self.headline or "").strip()
        if not text:
            raise ValueError("headline must not be empty or whitespace")
        self.headline = text
        return self


class ClickbaitAnalyzeResponse(BaseModel):
    is_clickbait: bool
    score: float = Field(ge=0.0, le=1.0)
    label: str
    confidence_note: Optional[str] = None
    contract_version: str
    detector_version: str
    evaluated_at: Optional[datetime] = None


class ClickbaitErrorResponse(BaseModel):
    code: str
    message: str
    errors: Optional[list[str]] = None
