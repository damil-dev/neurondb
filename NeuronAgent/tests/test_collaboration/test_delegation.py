"""Tests for Agent Delegation."""
import pytest

@pytest.mark.requires_server
class TestDelegation:
    def test_agent_delegation(self, api_client, test_agent, unique_name):
        """Test delegating tasks to specialized agents."""
        agent2 = api_client.post("/api/v1/agents", json_data={"name": unique_name, "system_prompt": "Specialized", "model_name": "gpt-4"})
        try:
            response = api_client.post(f"/api/v1/agents/{test_agent['id']}/delegate", json_data={"target_agent_id": agent2["id"], "task": "Test"})
            assert response is not None
        finally:
            api_client.delete(f"/api/v1/agents/{agent2['id']}")

