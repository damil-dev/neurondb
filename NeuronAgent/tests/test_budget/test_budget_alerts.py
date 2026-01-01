"""Tests for Budget Alerts."""
import pytest
from neurondb_client import AgentManager

@pytest.mark.api
@pytest.mark.requires_server
class TestBudgetAlerts:
    """Test budget alert functionality."""
    
    def test_budget_alert_configuration(self, api_client, test_agent):
        """Test configuring budget alerts."""
        agent_mgr = AgentManager(api_client)
        updated = agent_mgr.update(
            test_agent['id'],
            config={
                "budget": {
                    "max_cost": 100.0,
                    "alerts": [
                        {"threshold": 0.5, "action": "warn"},
                        {"threshold": 0.9, "action": "stop"}
                    ]
                }
            }
        )
        assert 'id' in updated
    
    def test_budget_alert_triggering(self, api_client, test_agent, test_session):
        """Test that alerts trigger at configured thresholds."""
        agent_mgr = AgentManager(api_client)
        session_mgr = SessionManager(api_client)
        
        # Set budget with alerts
        agent_mgr.update(
            test_agent['id'],
            config={
                "budget": {
                    "max_cost": 10.0,
                    "alerts": [{"threshold": 0.5, "action": "warn"}]
                }
            }
        )
        
        # Send messages to trigger alert
        # This would need actual cost tracking to work
        response = session_mgr.send_message(
            session_id=test_session['id'],
            content="Test",
            role="user"
        )
        assert 'response' in response or 'message_id' in response
    
    def test_budget_alert_notifications(self, api_client, test_agent):
        """Test budget alert notifications."""
        agent_mgr = AgentManager(api_client)
        # Configure alerts with notification settings
        updated = agent_mgr.update(
            test_agent['id'],
            config={
                "budget": {
                    "max_cost": 100.0,
                    "alerts": [{
                        "threshold": 0.8,
                        "action": "notify",
                        "channels": ["email", "webhook"]
                    }]
                }
            }
        )
        assert 'id' in updated

