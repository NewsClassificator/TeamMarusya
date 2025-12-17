import os
from pathlib import Path

# Repository root (backend/src/lib/clickbait_config.py -> backend/src -> backend -> repo)
REPO_ROOT = Path(__file__).resolve().parents[3]

CLICKBAIT_MODEL_PATH = Path(
    os.getenv("CLICKBAIT_MODEL_PATH", REPO_ROOT / "code/klikbait/my_awesome_model")
).resolve()
CLICKBAIT_MODULE_PATH = Path(
    os.getenv("CLICKBAIT_MODULE_PATH", REPO_ROOT / "code/klikbait/predict.py")
).resolve()
CLICKBAIT_THRESHOLD = float(os.getenv("CLICKBAIT_THRESHOLD", "0.5"))
CLICKBAIT_CONTRACT_VERSION = os.getenv("CLICKBAIT_CONTRACT_VERSION", "0.1.0")
CLICKBAIT_DETECTOR_VERSION = os.getenv("CLICKBAIT_DETECTOR_VERSION", "clickbait_model_v1")
