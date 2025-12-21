import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.lib.determinism import MODEL_VERSION


def _ensure_code_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    code_dir = repo_root / "code"
    if code_dir.is_dir():
        code_dir_str = str(code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)


_ensure_code_on_path()

from sentimen_analiz.main import RuBERTSentimentAnalyzer  # type: ignore  # noqa: E402


_analyzer: Optional[RuBERTSentimentAnalyzer] = None


def get_analyzer() -> RuBERTSentimentAnalyzer:
    """
    Lazily initialize the RuBERT sentiment analyzer using the fine-tuned model.
    """
    global _analyzer
    if _analyzer is None:
        repo_root = Path(__file__).resolve().parents[3]
        model_dir = repo_root / "code" / "sentimen_analiz" / "rubert_finetuned"
        model_name = str(model_dir)

        _analyzer = RuBERTSentimentAnalyzer(
            model_name=model_name,
            device="cpu",
            confidence_threshold=0.5,
        )
    return _analyzer


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Run sentiment analysis on the given text using the analyzer's chunking-aware API.
    """
    analyzer = get_analyzer()
    result = analyzer.predict_sentiment_with_chunking(text)
    return result


def map_label_to_contract(label: str) -> str:
    """
    Map model labels (NEGATIVE/NEUTRAL/POSITIVE/UNCERTAIN) to contract labels
    (negative/neutral/positive).
    """
    upper = label.upper()
    if upper == "NEGATIVE":
        return "negative"
    if upper == "POSITIVE":
        return "positive"
    # Both NEUTRAL and UNCERTAIN map to neutral in the contract
    return "neutral"


def get_model_version() -> str:
    """
    Return model version string used in API responses.
    Prefer explicit MODEL_VERSION from determinism module.
    """
    return MODEL_VERSION


def summarize_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment for a single text segment and map to contract fields.
    """
    raw = analyze_sentiment(text)
    return {
        "text": text,
        "sentiment_label": map_label_to_contract(raw.get("predicted_label", "NEUTRAL")),
        "confidence": float(raw.get("confidence", 0.0)),
    }


def analyze_sentiment_segments(main_text: str, quotes: list[str]) -> Dict[str, Any]:
    """
    Analyze sentiment for main text (with placeholders) and each quote individually.
    """
    main_summary = summarize_sentiment(main_text)

    quote_summaries = []
    for idx, quote_text in enumerate(quotes):
        summary = summarize_sentiment(quote_text)
        quote_summaries.append(
            {
                "quote_text": quote_text,
                "sentiment_label": summary["sentiment_label"],
                "confidence": summary["confidence"],
                "position": idx,
            }
        )

    return {
        "main_text": main_summary,
        "quotes": quote_summaries,
        "errors": [],
    }

