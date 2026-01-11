from fastapi import APIRouter

from src.api.schemas_water import WaterAnalyzeRequest, WaterAnalyzeResponse
from src.services.water_detector import analyze_water

router = APIRouter()


@router.post("/water-detection", response_model=WaterAnalyzeResponse, tags=["water"])
async def water_detection(payload: WaterAnalyzeRequest) -> WaterAnalyzeResponse:
    """
    Analyze text for water content and return label with confidence and optional feature signals.
    """
    return analyze_water(payload)
