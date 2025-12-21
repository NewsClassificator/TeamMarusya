from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_missing_both_url_and_text_returns_400() -> None:
    payload = {
        "input_type": "url",
        "language": "ru",
    }

    response = client.post("/analysis", json=payload)
    assert response.status_code == 400


def test_both_url_and_text_provided_returns_400() -> None:
    payload = {
        "input_type": "text",
        "url": "https://example.com",
        "text": "Текст",
        "language": "ru",
    }

    response = client.post("/analysis", json=payload)
    assert response.status_code == 400
