"""
API endpoint integration tests for server.py.
Validates POST /query contract, validation rules, and ensures non-query endpoints are not exposed.
"""

import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_query_endpoint_success():
    """Valid query returns top-k chunks with chunk_text, page_number, and score."""
    response = client.post(
        "/query",
        json={"query": "What are the four classes detected by the YOLOv8l model?", "top_k": 3}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3

    first = data[0]
    assert "chunk_text" in first and isinstance(first["chunk_text"], str)
    assert "page_number" in first and isinstance(first["page_number"], int)
    assert "score" in first and isinstance(first["score"], float)
    assert len(first["chunk_text"].strip()) > 0
    assert first["page_number"] >= 1


def test_query_empty_string_error():
    """Empty or whitespace queries must return HTTP 422 Unprocessable Entity."""
    response = client.post("/query", json={"query": "   ", "top_k": 3})
    assert response.status_code == 422


def test_query_invalid_top_k_error():
    """Non-positive top_k must return HTTP 422 validation error."""
    response = client.post("/query", json={"query": "valid query", "top_k": 0})
    assert response.status_code == 422

    response_neg = client.post("/query", json={"query": "valid query", "top_k": -5})
    assert response_neg.status_code == 422


def test_upload_endpoint_not_exposed():
    """Verify POST /upload does not exist and returns 404/405."""
    response = client.post("/upload")
    assert response.status_code in [404, 405]


def test_health_endpoint_not_exposed():
    """Verify GET /health is not exposed as an API endpoint (strict minimalism)."""
    response = client.get("/health")
    assert response.status_code in [404, 405]


def test_root_endpoint_not_exposed():
    """Verify GET / is not exposed on the API server (strict minimalism)."""
    response = client.get("/")
    assert response.status_code in [404, 405]


