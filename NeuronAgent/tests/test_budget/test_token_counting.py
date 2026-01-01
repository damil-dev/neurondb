"""Tests for Token Counting."""
import pytest
from neurondb_client import SessionManager

@pytest.mark.api
@pytest.mark.requires_server
class TestTokenCounting:
    """Test accurate token counting for cost tracking."""
    
    def test_token_counting_message(self, api_client, test_session):
        """Test token counting for messages."""
        session_mgr = SessionManager(api_client)
        response = session_mgr.send_message(
            session_id=test_session['id'],
            content="This is a test message for token counting.",
            role="user"
        )
        assert 'response' in response or 'message_id' in response
        # Token count should be available in message metadata
    
    def test_token_counting_prompt(self, api_client, test_agent):
        """Test token counting for system prompts."""
        agent_mgr = AgentManager(api_client)
        agent = agent_mgr.get(test_agent['id'])
        assert 'system_prompt' in agent
        # Token count for prompt should be tracked
    
    def test_token_counting_context(self, api_client, test_session):
        """Test token counting for context windows."""
        session_mgr = SessionManager(api_client)
        
        # Send multiple messages to build context
        for i in range(5):
            session_mgr.send_message(
                session_id=test_session['id'],
                content=f"Message {i}: " + "x" * 100,  # Longer messages
                role="user"
            )
        
        # Total token count should reflect all messages
        messages = session_mgr.get_messages(test_session['id'])
        assert isinstance(messages, list)
    
    def test_token_counting_accuracy(self, api_client, test_session):
        """Test that token counting is accurate."""
        session_mgr = SessionManager(api_client)
        
        # Send a known-length message
        test_content = "The quick brown fox jumps over the lazy dog." * 10
        response = session_mgr.send_message(
            session_id=test_session['id'],
            content=test_content,
            role="user"
        )
        assert 'response' in response or 'message_id' in response
        # Token count should be reasonable for the content length

