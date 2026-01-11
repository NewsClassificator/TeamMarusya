import importlib.util
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict

from fastapi import HTTPException, status

from src.api.schemas_water import (
    FeatureMetrics,
    WaterAnalyzeRequest,
    WaterAnalyzeResponse,
)
from src.lib.determinism import create_determinism_context
from src.lib.water_config import (
    WATER_CONTRACT_VERSION,
    WATER_DETECTOR_VERSION,
    WATER_MODEL_PATH,
    WATER_MODULE_PATH,
)


def _load_analyzer_module():
    if not WATER_MODULE_PATH.exists():
        raise RuntimeError(f"Water analyzer module not found at {WATER_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("water_analyzer", WATER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load water analyzer module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@lru_cache(maxsize=1)
def _get_analyzer():
    module = _load_analyzer_module()
    analyzer_cls = getattr(module, "WaterAnalyzer", None)
    if analyzer_cls is None:
        raise RuntimeError("WaterAnalyzer class not found in analyzer module")
    return analyzer_cls(model_path=str(WATER_MODEL_PATH))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _map_features(raw: Dict[str, Any] | None) -> FeatureMetrics | None:
    if not raw:
        return None
    return FeatureMetrics(
        readability_index=_safe_float(raw.get("readability_index")),
        adj_ratio=_safe_float(raw.get("adj_ratio")),
        adv_ratio=_safe_float(raw.get("adv_ratio")),
        repetition_ratio=_safe_float(raw.get("repetition_ratio")),
        stopword_ratio=_safe_float(raw.get("stopword_ratio"))
        if raw.get("stopword_ratio") is not None
        else None,
    )


def _fallback_response(message: str) -> WaterAnalyzeResponse:
    now = datetime.now(timezone.utc)
    return WaterAnalyzeResponse(
        is_water=False,
        label="status unavailable",
        confidence=0.0,
        water_percentage=0.0,
        features=None,
        interpretations=None,
        contract_version=WATER_CONTRACT_VERSION,
        detector_version=WATER_DETECTOR_VERSION,
        evaluated_at=now,
        errors=[message],
    )


def analyze_water(payload: WaterAnalyzeRequest) -> WaterAnalyzeResponse:
    """
    Run water detection for a single text sample. Returns structured response.
    """
    create_determinism_context()

    try:
        analyzer = _get_analyzer()
    except Exception as exc:  # pragma: no cover - defensive path
        return _fallback_response(f"water detector init error: {exc}")

    try:
        result = analyzer.analyze(payload.text, detailed=payload.include_features)
    except Exception as exc:
        return _fallback_response(f"water detector unavailable: {exc}")

    features = _map_features(result.get("features") if payload.include_features else None)
    interpretations = (
        result.get("interpretations") if payload.include_features else None
    )

    response = WaterAnalyzeResponse(
        is_water=bool(result.get("is_water")),
        label=str(result.get("water_label") or ""),
        confidence=_safe_float(result.get("confidence")),
        water_percentage=_safe_float(result.get("water_percentage")),
        features=features,
        interpretations=interpretations,
        contract_version=WATER_CONTRACT_VERSION,
        detector_version=WATER_DETECTOR_VERSION,
        evaluated_at=datetime.now(timezone.utc),
    )

    # Ensure percentage present for clients expecting explicit percent
    if response.water_percentage is None:
        response.water_percentage = response.confidence * 100

    return response
