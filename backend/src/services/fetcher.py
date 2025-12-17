import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def _ensure_code_on_path() -> None:
    """
    Ensure that the top-level `code/` directory is importable when running
    the backend as an isolated project.
    Also load the local backend/.env so Oxylabs credentials can be configured there.
    """
    # backend/ directory
    backend_root = Path(__file__).resolve().parents[2]

    # Load backend/.env
    env_path = backend_root / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)

    # Add /code so we can import the sentiment and parser projects.
    repo_root = backend_root.parent
    code_dir = repo_root / "code"
    if code_dir.is_dir():
        code_dir_str = str(code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)

    # Also add /code/parser so that `site_parsers` (used inside parser/main.py)
    # can be imported the same way as in the original project layout.
    parser_dir = code_dir / "parser"
    if parser_dir.is_dir():
        parser_dir_str = str(parser_dir)
        if parser_dir_str not in sys.path:
            sys.path.insert(0, parser_dir_str)


_ensure_code_on_path()

from parser.main import NewsParser  # type: ignore  # noqa: E402


class FetchError(Exception):
    pass


def get_news_parser() -> NewsParser:
    """
    Create a NewsParser instance using Oxylabs credentials from environment.
    """
    username = os.getenv("OXYLABS_USERNAME")
    password = os.getenv("OXYLABS_PASSWORD")

    if not username or not password:
        raise FetchError(
            "Oxylabs credentials are not configured. "
            "Set OXYLABS_USERNAME and OXYLABS_PASSWORD environment variables."
        )

    return NewsParser(username=username, password=password)


def fetch_article(url: str, debug: bool = False) -> Dict[str, Optional[str]]:
    """
    Fetch and parse article information using the existing NewsParser.

    Returns a dict with keys: title, text, date, author, url, parser_type, error.
    """
    parser = get_news_parser()
    result: Dict[str, Any] = parser.get_news_info(url=url, debug=debug)

    # Normalize keys we care about; keep unknown keys for potential debugging.
    return {
        "title": result.get("title"),
        "text": result.get("text"),
        "date": result.get("date"),
        "author": result.get("author"),
        "url": result.get("url", url),
        "parser_type": result.get("parser_type"),
        "error": result.get("error"),
    }
