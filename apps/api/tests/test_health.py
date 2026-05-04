import pytest
from fastapi.testclient import TestClient

from equityiq_api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_thesis_validates_ticker(client: TestClient) -> None:
    resp = client.post("/thesis/stream", json={"ticker": "lowercase", "question": "what?"})
    assert resp.status_code == 422


def test_thesis_validates_question_length(client: TestClient) -> None:
    resp = client.post("/thesis/stream", json={"ticker": "NVDA", "question": "hi"})
    assert resp.status_code == 422
