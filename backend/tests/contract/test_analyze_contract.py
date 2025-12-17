from fastapi.testclient import TestClient

from src.api.app import app
from src.services import analyzer as analyzer_module
from src.services import fetcher as fetcher_module


client = TestClient(app)


def test_health_endpoint_available() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_analyze_text_happy_path(monkeypatch) -> None:
    """
    Happy-path contract test for /analyze with text input.
    We stub sentiment so the test does not depend on heavy model loading.
    """

    def fake_analyze_sentiment(text: str):
        return {"predicted_label": "POSITIVE", "confidence": 0.95}

    monkeypatch.setattr(
        analyzer_module, "analyze_sentiment", fake_analyze_sentiment, raising=True
    )

    payload = {
        "input_type": "text",
        "text": "Это очень хорошая новость!",
        "language": "ru",
        "request_id": "test-text-1",
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["request_id"] == "test-text-1"
    assert data["contract_version"]
    assert data["model_version"]
    assert isinstance(data["seed"], int)

    assert data["article"]["content"]
    assert data["sentiment"]["label"] == "positive"
    assert data["sentiment"]["score"] == 0.95


def test_analyze_url_happy_path(monkeypatch) -> None:
    """
    Happy-path contract test for /analyze with URL input.
    We stub both fetcher and sentiment to avoid network and heavy model.
    """

    def fake_fetch_article(url: str, debug: bool = False):
        return {
            "title": "Заголовок новости",
            "text": "Текст новости",
            "date": "2025-01-01T00:00:00Z",
            "author": "Автор",
            "url": url,
            "parser_type": "specialized",
            "error": None,
        }

    def fake_analyze_sentiment(text: str):
        return {"predicted_label": "NEGATIVE", "confidence": 0.8}

    monkeypatch.setattr(
        fetcher_module, "fetch_article", fake_fetch_article, raising=True
    )
    monkeypatch.setattr(
        analyzer_module, "analyze_sentiment", fake_analyze_sentiment, raising=True
    )

    payload = {
        "input_type": "url",
        "url": "https://example.com/news/1",
        "language": "ru",
        "request_id": "test-url-1",
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["request_id"] == "test-url-1"
    assert data["article"]["title"] == "Заголовок новости"
    assert data["article"]["content"] == "Текст новости"
    assert data["article"]["author"] == "Автор"
    assert data["article"]["published_at"] == "2025-01-01T00:00:00Z"

    assert data["sentiment"]["label"] == "negative"
    assert data["sentiment"]["score"] == 0.8

