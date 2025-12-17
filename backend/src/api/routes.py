from fastapi import APIRouter

from src.api.schemas import AnalyzeRequest, AnalyzeResponse
from src.services.analyzer import analyze_request


router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_endpoint(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze news content from URL or raw text and return structured article
    fields and sentiment.
    """
    return analyze_request(payload)


