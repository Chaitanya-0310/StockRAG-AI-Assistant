import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    with patch('backend.services.db_service.init_db', new_callable=AsyncMock):
        from backend.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_root_returns_ok(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "StockRAG API is running"


class TestSymbolValidation:
    def test_valid_symbol(self, client):
        """Valid symbols should not return 400."""
        # We'll get 500 because DB isn't connected, but NOT 400
        response = client.get("/stock/AAPL")
        assert response.status_code != 400

    def test_invalid_symbol_too_long(self, client):
        response = client.get("/stock/TOOLONG")
        assert response.status_code == 400

    def test_invalid_symbol_lowercase(self, client):
        # lowercase gets uppercased, so this should be valid
        response = client.get("/stock/aapl")
        assert response.status_code != 400

    def test_invalid_symbol_numbers(self, client):
        response = client.get("/stock/123")
        assert response.status_code == 400

    def test_invalid_symbol_special_chars(self, client):
        response = client.get("/stock/A@PL")
        assert response.status_code == 400


class TestChatEndpoint:
    def test_chat_requires_body(self, client):
        response = client.post("/chat")
        assert response.status_code == 422  # Validation error


class TestPipelineStatus:
    def test_pipeline_status_endpoint_exists(self, client):
        """Pipeline status endpoint should exist (may fail if DB not connected)."""
        response = client.get("/pipeline/status")
        # Should not be 404
        assert response.status_code != 404
