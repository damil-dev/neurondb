# NeuronDB Python SDK

Python client libraries for NeuronDB ecosystem components.

## Installation

```bash
pip install neuronagent
# Or from source
pip install -e ./neuronagent
```

## Usage

### NeuronAgent

```python
from neuronagent import NeuronAgentClient

# Initialize client
client = NeuronAgentClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# Create an agent
agent = client.agents.create_agent(
    name="my-agent",
    system_prompt="You are a helpful assistant",
    model_name="gpt-4",
    enabled_tools=["sql", "http"]
)

# Create a session
session = client.sessions.create_session(agent_id=agent.id)

# Send a message
response = client.sessions.send_message(
    session_id=session.id,
    content="What is the weather today?"
)

print(response.content)
```

### NeuronDB (Direct SQL)

```python
import psycopg2
from neurondb import NeuronDBClient

# Connect to NeuronDB
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="neurondb",
    user="neurondb",
    password="neurondb"
)

# Use NeuronDB functions
with conn.cursor() as cur:
    # Vector search
    cur.execute("""
        SELECT id, content, embedding <=> %s AS distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT 10
    """, (query_vector, query_vector))
    
    results = cur.fetchall()
```

## Examples

See `examples/` directory for complete examples:
- `basic_agent.py` - Basic agent usage
- `rag_pipeline.py` - RAG pipeline example
- `vector_search.py` - Vector search example

## API Reference

Full API documentation is available at:
- NeuronAgent: https://www.neurondb.ai/docs/neuronagent/api





