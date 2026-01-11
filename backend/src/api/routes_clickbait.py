from fastapi import APIRouter

from src.api.schemas_clickbait import ClickbaitAnalyzeRequest, ClickbaitAnalyzeResponse
from src.services.clickbait_detector import analyze_clickbait

router = APIRouter()


@router.post("/clickbait/analyze", response_model=ClickbaitAnalyzeResponse, tags=["clickbait"])
async def clickbait_analyze_endpoint(payload: ClickbaitAnalyzeRequest) -> ClickbaitAnalyzeResponse:
    """
    Evaluate a headline and return clickbait status with confidence data.
    """
    return analyze_clickbait(payload)
