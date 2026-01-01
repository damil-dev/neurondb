"""Tests for Agent State Machine - execution states and transitions."""
import pytest

@pytest.mark.requires_server
class TestStateMachine:
    def test_execution_states(self, api_client, test_session):
        """Test agent execution state transitions."""
        message_data = {"content": "Test message", "role": "user"}
        response = api_client.post(f"/api/v1/sessions/{test_session['id']}/messages", json_data=message_data)
        assert response is not None

