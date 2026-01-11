# NeuronMCP Python SDK

A comprehensive Python SDK for interacting with NeuronMCP (Model Context Protocol) server.

## Features

- ✅ Full MCP protocol support
- ✅ Async/await support
- ✅ Type hints throughout
- ✅ Comprehensive error handling
- ✅ Automatic retry logic
- ✅ Connection pooling
- ✅ Request/response logging

## Installation

```bash
pip install neurondb-mcp-python
```

## Quick Start

```python
import asyncio
from neurondb_mcp import NeuronMCPClient

async def main():
    async with NeuronMCPClient("http://localhost:8080", api_key="your-api-key") as client:
        # List all available tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)}")
        
        # Call a tool
        result = await client.call_tool("vector_search", {
            "table": "documents",
            "vector_column": "embedding",
            "query_vector": [0.1, 0.2, 0.3],
            "limit": 10
        })
        
        print(f"Results: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Examples

See the `examples/` directory for more comprehensive examples.

## Documentation

Full documentation available at: https://docs.neurondb.com/mcp/python-sdk

## License

MIT License
