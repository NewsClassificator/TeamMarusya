import importlib.util
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from fastapi import HTTPException, status

from src.api.schemas_clickbait import ClickbaitAnalyzeRequest, ClickbaitAnalyzeResponse
from src.lib.clickbait_config import (
    CLICKBAIT_CONTRACT_VERSION,
    CLICKBAIT_DETECTOR_VERSION,
    CLICKBAIT_MODEL_PATH,
    CLICKBAIT_MODULE_PATH,
    CLICKBAIT_THRESHOLD,
)
from src.lib.determinism import create_determinism_context


def _load_predict_module():
    if not CLICKBAIT_MODULE_PATH.exists():
        raise RuntimeError(f"Clickbait module not found at {CLICKBAIT_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("klikbait_predict", CLICKBAIT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load clickbait predictor module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@lru_cache(maxsize=1)
def _get_detector():
    module = _load_predict_module()
    detector_cls = getattr(module, "ClickbaitDetector", None)
    if detector_cls is None:
        raise RuntimeError("ClickbaitDetector class not found in predict module")
    return detector_cls(model_path=str(CLICKBAIT_MODEL_PATH))


def _normalize_score(raw_score: Any) -> float:
    try:
        score = float(raw_score)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _confidence_note(score: float) -> str | None:
    margin = abs(score - CLICKBAIT_THRESHOLD)
    if margin <= 0.1:
        return "Низкая уверенность: результат близок к пороговому значению."
    return None


def _fallback_response(message: str) -> ClickbaitAnalyzeResponse:
    now = datetime.now(timezone.utc)
    return ClickbaitAnalyzeResponse(
        is_clickbait=False,
        score=0.0,
        label="status unavailable",
        confidence_note=message,
        contract_version=CLICKBAIT_CONTRACT_VERSION,
        detector_version=CLICKBAIT_DETECTOR_VERSION,
        evaluated_at=now,
    )


def analyze_clickbait(payload: ClickbaitAnalyzeRequest) -> ClickbaitAnalyzeResponse:
    """
    Run clickbait detection for a single headline. Returns a structured response or
    a neutral fallback when inference fails.
    """
    create_determinism_context()  # sets seeds for deterministic scoring

    try:
        detector = _get_detector()
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "CLICKBAIT_INIT_ERROR", "message": str(exc)},
        ) from exc

    try:
        result = detector.predict(payload.headline)
    except Exception as exc:
        # Graceful neutral fallback while preserving API contract
        return _fallback_response(f"clickbait detector unavailable: {exc}")

    score = _normalize_score(result.get("score"))
    is_clickbait = detector.is_clickbait(payload.headline, threshold=CLICKBAIT_THRESHOLD)
    label = "clickbait" if is_clickbait else "not clickbait"
    note = _confidence_note(score)

    return ClickbaitAnalyzeResponse(
        is_clickbait=is_clickbait,
        score=score,
        label=label,
        confidence_note=note,
        contract_version=CLICKBAIT_CONTRACT_VERSION,
        detector_version=CLICKBAIT_DETECTOR_VERSION,
        evaluated_at=datetime.now(timezone.utc),
    )
