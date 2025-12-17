import os
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def configure_cpu_only_environment() -> Generator[None, None, None]:
    """
    Ensure tests run in CPU-only deterministic mode where possible.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    yield

