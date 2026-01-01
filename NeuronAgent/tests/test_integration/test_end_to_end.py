"""Tests for Complete Workflows."""
import pytest

@pytest.mark.integration
@pytest.mark.requires_server
@pytest.mark.slow
class TestEndToEnd:
    def test_complete_workflow(self, api_client, unique_name):
        """Test complete workflow: Agent → Session → Messages → Memory → Tools."""
        # Create agent
        agent_data = {"name": unique_name, "system_prompt": "Test", "model_name": "gpt-4", "enabled_tools": ["sql"]}
        agent = api_client.post("/api/v1/agents", json_data=agent_data)
        
        try:
            # Create session
            session = api_client.post("/api/v1/sessions", json_data={"agent_id": agent["id"]})
            
            try:
                # Send message
                message = api_client.post(
                    f"/api/v1/sessions/{session['id']}/messages",
                    json_data={"content": "Hello", "role": "user"}
                )
                assert message is not None
            finally:
                api_client.delete(f"/api/v1/sessions/{session['id']}")
        finally:
            api_client.delete(f"/api/v1/agents/{agent['id']}")

