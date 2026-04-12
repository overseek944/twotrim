"""Tests for the interceptor proxy."""

import pytest
from fastapi.testclient import TestClient
from twotrim.interceptor.proxy import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestProxyEndpoints:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_stats(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "total_tokens_saved" in data

    def test_recent_stats(self, client):
        resp = client.get("/stats/recent?n=10")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_chat_completions_missing_body(self, client):
        resp = client.post("/v1/chat/completions", content=b"invalid")
        assert resp.status_code == 400

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
