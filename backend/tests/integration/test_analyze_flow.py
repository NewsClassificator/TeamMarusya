from fastapi.testclient import TestClient

from src.api.app import app
from src.services import analyzer as analyzer_module


client = TestClient(app)


def test_analyze_text_integration_with_stubbed_sentiment(monkeypatch) -> None:
    """
    Integration-like test that exercises the full FastAPI stack for text input,
    while stubbing out the heavy sentiment model.
    """

    def fake_analyze_sentiment(text: str):
        return {"predicted_label": "NEUTRAL", "confidence": 0.7}

    monkeypatch.setattr(
        analyzer_module, "analyze_sentiment", fake_analyze_sentiment, raising=True
    )

    payload = {
        "input_type": "text",
        "text": "Новости без ярко выраженной эмоции.",
        "language": "ru",
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["article"]["content"]
    assert data["sentiment"]["label"] == "neutral"
    assert data["sentiment"]["score"] == 0.7

