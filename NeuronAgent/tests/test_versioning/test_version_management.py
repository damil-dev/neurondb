"""Tests for Version Management."""
import pytest

@pytest.mark.requires_server
class TestVersionManagement:
    def test_version_management(self, api_client, test_agent):
        """Test version control for agents."""
        response = api_client.get(f"/api/v1/agents/{test_agent['id']}/versions")
        assert isinstance(response, (list, dict))

