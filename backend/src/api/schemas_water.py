from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from src.lib.water_config import TEXT_MAX_LENGTH, TEXT_MIN_LENGTH


class FeatureMetrics(BaseModel):
    readability_index: float = Field(default=0.0)
    adj_ratio: float = Field(default=0.0)
    adv_ratio: float = Field(default=0.0)
    repetition_ratio: float = Field(default=0.0)
    stopword_ratio: Optional[float] = Field(default=None, description="Deprecated; may be absent.")


class WaterAnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for water content.")
    include_features: bool = Field(default=True, description="Return feature metrics and interpretations.")
    language: Optional[str] = Field(default=None, description="Optional language hint for messaging; not used for routing.")

    @model_validator(mode="after")
    def validate_text(self) -> "WaterAnalyzeRequest":
        content = (self.text or "").strip()
        if not content:
            raise ValueError("text must not be empty or whitespace")
        if len(content) < TEXT_MIN_LENGTH:
            raise ValueError(f"text must be at least {TEXT_MIN_LENGTH} characters")
        if len(content) > TEXT_MAX_LENGTH:
            raise ValueError(f"text must be at most {TEXT_MAX_LENGTH} characters")
        if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in content):
            raise ValueError("text contains unsupported control characters")
        self.text = content
        return self


class WaterAnalyzeResponse(BaseModel):
    is_water: bool
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    water_percentage: Optional[float] = Field(default=None, description="Water probability in percent (0-100).")
    features: Optional[FeatureMetrics] = None
    interpretations: Optional[Dict[str, str]] = None
    contract_version: str
    detector_version: str
    evaluated_at: Optional[datetime] = None
    errors: Optional[List[str]] = None


class WaterErrorResponse(BaseModel):
    code: str
    message: str
    errors: Optional[List[str]] = None
