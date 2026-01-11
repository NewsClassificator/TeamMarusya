import argparse
import json
from typing import Any, Dict

from src.api.schemas import AnalyzeRequest
from src.services.analyzer import analyze_request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run news analysis from CLI")
    parser.add_argument(
        "--url",
        type=str,
        help="URL of the news article to analyze",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Raw text to analyze (if no URL is provided)",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Optional request identifier",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.url and args.text:
        raise SystemExit("Provide either --url or --text, not both.")

    if args.url:
        input_type = "url"
    elif args.text:
        input_type = "text"
    else:
        raise SystemExit("You must provide either --url or --text.")

    payload = AnalyzeRequest(
        input_type=input_type,
        url=getattr(args, "url", None),
        text=getattr(args, "text", None),
        request_id=args.request_id,
    )

    result = analyze_request(payload)
    as_dict: Dict[str, Any] = json.loads(result.model_dump_json())
    print(json.dumps(as_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

