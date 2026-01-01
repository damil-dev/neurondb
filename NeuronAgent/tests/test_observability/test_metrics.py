"""Tests for Prometheus Metrics."""
import pytest
import requests

@pytest.mark.requires_server
class TestMetrics:
    def test_prometheus_metrics(self):
        """Test Prometheus metrics export."""
        response = requests.get("http://localhost:8080/metrics", timeout=5)
        assert response.status_code == 200
        assert "prometheus" in response.text.lower() or len(response.text) > 0

