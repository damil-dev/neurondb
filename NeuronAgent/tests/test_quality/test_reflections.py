"""Tests for Reflections."""
import pytest

@pytest.mark.requires_server
class TestReflections:
    def test_reflections(self, api_client, test_session):
        """Test agent self-reflection."""
        response = api_client.get(f"/api/v1/sessions/{test_session['id']}/reflections")
        assert isinstance(response, (list, dict))

