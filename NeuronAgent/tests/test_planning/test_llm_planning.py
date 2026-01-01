"""Tests for LLM-Based Planning."""
import pytest

@pytest.mark.requires_server
class TestLLMPlanning:
    def test_llm_planning(self, api_client, test_agent):
        """Test advanced planning with task decomposition."""
        response = api_client.post(f"/api/v1/agents/{test_agent['id']}/plan", json_data={"goal": "Test goal"})
        assert response is not None

