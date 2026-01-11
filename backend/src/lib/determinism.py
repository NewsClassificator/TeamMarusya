import os
import random
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is optional
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore


CONTRACT_VERSION = "0.1.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "rubert_finetuned_v1")


@dataclass
class DeterminismContext:
    seed: int
    contract_version: str = CONTRACT_VERSION
    model_version: str = MODEL_VERSION


def set_seed(seed: int) -> None:
    """
    Configure deterministic random seed for Python, NumPy and torch (if available).
    """
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on system
            torch.cuda.manual_seed_all(seed)


def create_determinism_context(seed: Optional[int] = None) -> DeterminismContext:
    """
    Create and apply a deterministic context, returning the values
    that should be echoed in API responses.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    set_seed(seed)
    return DeterminismContext(seed=seed)

