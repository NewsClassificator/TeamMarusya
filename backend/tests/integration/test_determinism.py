from fastapi.testclient import TestClient

from src.api.app import app
from src.services import analyzer as analyzer_module


client = TestClient(app)


def test_repeated_requests_produce_identical_sentiment(monkeypatch) -> None:
    """
    Determinism smoke test: with a stubbed sentiment function that uses
    only deterministic logic, repeated requests should yield identical
    label/score for the same input.
    """

    def fake_analyze_sentiment(text: str):
        # Simple deterministic mapping based on text length parity
        if len(text) % 2 == 0:
          return {"predicted_label": "POSITIVE", "confidence": 0.9}
        return {"predicted_label": "NEGATIVE", "confidence": 0.8}

    monkeypatch.setattr(
        analyzer_module, "analyze_sentiment", fake_analyze_sentiment, raising=True
    )

    payload = {
        "input_type": "text",
        "text": "Одинаковый текст для проверки детерминизма.",
        "language": "ru",
    }

    first = client.post("/analyze", json=payload)
    second = client.post("/analyze", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    first_data = first.json()
    second_data = second.json()

    assert first_data["sentiment"] == second_data["sentiment"]

