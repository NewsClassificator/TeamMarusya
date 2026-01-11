from fastapi.testclient import TestClient

from src.api.app import app
from src.services import analyzer as analyzer_module


client = TestClient(app)


def test_analyze_text_integration_with_stubbed_sentiment(monkeypatch) -> None:
    """
    Integration-like test that exercises the full FastAPI stack for text input,
    while stubbing out the heavy sentiment model.
    """

    def fake_analyze_sentiment_segments(main_text: str, quotes: list[str]):
        return {
            "main_text": {
                "text": main_text,
                "sentiment_label": "neutral",
                "confidence": 0.7,
            },
            "quotes": [
                {
                    "quote_text": q,
                    "sentiment_label": "positive",
                    "confidence": 0.6,
                    "position": i,
                    "author": "Автор",
                }
                for i, q in enumerate(quotes)
            ],
            "errors": [],
        }

    monkeypatch.setattr(
        analyzer_module, "analyze_sentiment_segments", fake_analyze_sentiment_segments, raising=True
    )

    payload = {
        "input_type": "text",
        "text": 'Новости без ярко выраженной эмоции. "Цитата"',
        "language": "ru",
    }

    response = client.post("/analysis", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["article"]["content"]
    assert data["sentiment"]["main_text"]["sentiment_label"] == "neutral"
    assert data["sentiment"]["main_text"]["confidence"] == 0.7
    assert data["sentiment"]["quotes"][0]["quote_text"] == "Цитата"
