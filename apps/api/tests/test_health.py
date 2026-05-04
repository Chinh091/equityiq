import pytest
from equityiq_api import app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# /thesis/stream validation is covered in test_thesis_stream.py with the
# agent_loop dependency override.
