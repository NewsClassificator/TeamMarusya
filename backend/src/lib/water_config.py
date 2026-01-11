import os
from pathlib import Path

# Repository root (backend/src/lib/water_config.py -> backend/src -> backend -> repo)
REPO_ROOT = Path(__file__).resolve().parents[3]

WATER_MODEL_PATH = Path(
    os.getenv("WATER_MODEL_PATH", REPO_ROOT / "code/water/ruber_quality_model.pkl")
).resolve()
WATER_MODULE_PATH = Path(
    os.getenv("WATER_MODULE_PATH", REPO_ROOT / "code/water/water_analyzer.py")
).resolve()

WATER_CONTRACT_VERSION = os.getenv("WATER_CONTRACT_VERSION", "1.0.0")
WATER_DETECTOR_VERSION = os.getenv("WATER_DETECTOR_VERSION", "water-detector-1.0")

TEXT_MIN_LENGTH = int(os.getenv("WATER_TEXT_MIN_LENGTH", "20"))
TEXT_MAX_LENGTH = int(os.getenv("WATER_TEXT_MAX_LENGTH", "10000"))
