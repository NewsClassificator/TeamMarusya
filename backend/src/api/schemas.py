from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    input_type: Literal["url", "text"]
    url: Optional[str] = Field(
        default=None,
        description="Required when input_type=url",
    )
    text: Optional[str] = Field(
        default=None,
        description="Required when input_type=text",
    )
    language: str = Field(default="ru")
    request_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_input(self) -> "AnalyzeRequest":
        """
        Enforce that exactly one of {url, text} is provided and
        matches the declared input_type. Otherwise FastAPI will
        return a 400 validation error.
        """
        if self.input_type == "url":
            if not self.url or self.text is not None:
                raise ValueError("For input_type='url' provide only 'url' field")
        elif self.input_type == "text":
            if not self.text or self.url is not None:
                raise ValueError("For input_type='text' provide only 'text' field")
        else:  # pragma: no cover - protected by Literal
            raise ValueError("input_type must be 'url' or 'text'")

        return self


class ArticleContent(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    content: str


class SentimentResult(BaseModel):
    label: Literal["positive", "neutral", "negative"]
    score: float


class AnalyzeResponse(BaseModel):
    request_id: Optional[str] = None
    contract_version: str
    model_version: str
    seed: int
    article: ArticleContent
    sentiment: SentimentResult


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None

