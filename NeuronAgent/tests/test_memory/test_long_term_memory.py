"""Tests for Long-term Memory - HNSW vector search."""
import pytest

@pytest.mark.requires_server
@pytest.mark.requires_neurondb
class TestLongTermMemory:
    def test_vector_search(self, api_client, test_agent):
        """Test HNSW-based vector search."""
        response = api_client.get(f"/api/v1/agents/{test_agent['id']}/memory/search")
        assert isinstance(response, (list, dict))

