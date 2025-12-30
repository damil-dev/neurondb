# NeuronDB Examples

**Complete examples and tutorials for building AI applications with NeuronDB**

This directory contains practical, production-ready examples demonstrating how to use NeuronDB ecosystem components for common AI application patterns.

---

## 📚 Available Examples

### 1. Semantic Search on Documents

**Location:** [`semantic-search-docs/`](semantic-search-docs/)

Build a semantic search engine over document collections using vector embeddings.

**What you'll learn:**
- Generate embeddings for text documents
- Create and optimize HNSW indexes
- Perform similarity search with different distance metrics
- Combine semantic and keyword search (hybrid search)

**Technologies:** NeuronDB vector search, embeddings, HNSW indexing

**Time to complete:** 20 minutes

---

### 2. RAG Chatbot with PDFs

**Location:** [`rag-chatbot-pdfs/`](rag-chatbot-pdfs/)

Create a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on PDF documents.

**What you'll learn:**
- Extract text from PDF documents
- Chunk documents for optimal RAG performance
- Generate embeddings and store in vector database
- Implement RAG pipeline with context retrieval
- Integrate with LLMs for response generation

**Technologies:** NeuronDB RAG pipeline, NeuronAgent, document processing

**Time to complete:** 30 minutes

---

### 3. AI Agent Tools

**Location:** [`agent-tools/`](agent-tools/)

Build custom tools for AI agents using NeuronAgent's tool framework.

**What you'll learn:**
- Create custom agent tools
- Register tools with NeuronAgent
- Implement tool validation and error handling
- Build multi-tool workflows
- Use agent long-term memory

**Technologies:** NeuronAgent REST API, tool registry, WebSocket streaming

**Time to complete:** 25 minutes

---

### 4. MCP Integration

**Location:** [`mcp-integration/`](mcp-integration/)

Integrate NeuronDB with MCP-compatible clients like Claude Desktop.

**What you'll learn:**
- Configure NeuronMCP server
- Connect Claude Desktop to NeuronDB
- Use MCP tools from AI assistants
- Build custom MCP workflows
- Access PostgreSQL through MCP protocol

**Technologies:** NeuronMCP, Model Context Protocol, Claude Desktop

**Time to complete:** 15 minutes

---

## 🎯 Example Categories

### By Use Case

| Use Case | Example | Complexity |
|----------|---------|------------|
| **Document Search** | Semantic Search | ⭐ Beginner |
| **Question Answering** | RAG Chatbot | ⭐⭐ Intermediate |
| **Agent Automation** | Agent Tools | ⭐⭐ Intermediate |
| **AI Assistant Integration** | MCP Integration | ⭐ Beginner |

### By Component

| Component | Examples Using It |
|-----------|------------------|
| **NeuronDB** | All examples (core database) |
| **NeuronAgent** | RAG Chatbot, Agent Tools |
| **NeuronMCP** | MCP Integration |
| **NeuronDesktop** | (Use any example through UI) |

### By Technology

| Technology | Examples |
|------------|----------|
| **Vector Search** | Semantic Search, RAG Chatbot |
| **Embeddings** | Semantic Search, RAG Chatbot |
| **ML Models** | RAG Chatbot |
| **RAG Pipeline** | RAG Chatbot |
| **Agent Runtime** | Agent Tools, RAG Chatbot |
| **MCP Protocol** | MCP Integration |

---

## 🚀 Getting Started

### Prerequisites

1. **NeuronDB Running:** Follow [QUICKSTART.md](../QUICKSTART.md)
2. **Components:** Install components needed for your example
3. **API Keys:** Some examples require OpenAI/Anthropic API keys

### Quick Setup

```bash
# Start all services
docker compose up -d

# Verify services are healthy
docker compose ps

# Navigate to an example
cd examples/semantic-search-docs/
```

---

## 📖 Example Structure

Each example follows a consistent structure:

```
example-name/
├── README.md           # Complete tutorial and documentation
├── setup.sql          # Database schema and setup (if applicable)
├── data/              # Sample data files
├── src/               # Source code
│   ├── main.*         # Main implementation
│   └── utils.*        # Utility functions
├── config/            # Configuration files
└── tests/             # Example tests
```

---

## 💡 Tips for Using Examples

### Best Practices

1. **Read README First:** Each example has detailed instructions
2. **Start Simple:** Begin with beginner examples
3. **Understand Concepts:** Focus on learning patterns, not just copying code
4. **Experiment:** Modify examples to fit your use case
5. **Check Prerequisites:** Ensure required services are running

### Common Patterns

All examples demonstrate these key patterns:

- **Database Connection:** How to connect to NeuronDB
- **Error Handling:** Proper error handling and validation
- **Configuration:** Environment-based configuration
- **Testing:** How to test your implementation
- **Deployment:** Production deployment considerations

---

## 🎓 Learning Path

### Beginner Path

1. Start with **Semantic Search** to learn vector basics
2. Try **MCP Integration** to understand the ecosystem
3. Build confidence with simple queries and operations

### Intermediate Path

1. Complete beginner examples first
2. Move to **RAG Chatbot** for full-stack RAG
3. Explore **Agent Tools** for custom automation
4. Combine patterns in your own projects

### Advanced Topics

After completing examples, explore:

- Custom ML models ([NeuronDB ML docs](../NeuronDB/docs/ml-algorithms/))
- Advanced indexing ([Indexing guide](../NeuronDB/docs/vector-search/indexing.md))
- Production optimization ([Performance docs](../NeuronDB/docs/performance/))
- Multi-modal embeddings ([Embedding docs](../NeuronDB/docs/ml-embeddings/))

---

## 🔗 Additional Resources

### Documentation

- **[Main README](../README.md)** - Project overview
- **[Quick Start](../QUICKSTART.md)** - Get started in minutes
- **[Documentation Index](../DOCUMENTATION.md)** - Complete documentation reference

### Component Documentation

- **[NeuronDB Docs](../NeuronDB/README.md)** - Database extension
- **[NeuronAgent Docs](../NeuronAgent/README.md)** - Agent runtime
- **[NeuronMCP Docs](../NeuronMCP/README.md)** - MCP server
- **[NeuronDesktop Docs](../NeuronDesktop/README.md)** - Web interface

### More Code Examples

- **[NeuronDB Demo](../NeuronDB/demo/)** - 60+ SQL examples
- **[NeuronAgent Examples](../NeuronAgent/examples/)** - 38 Python/Go examples
- **[NeuronMCP Examples](../NeuronMCP/docs/examples/)** - MCP client examples

---

## 🤝 Contributing Examples

Have a great example to share? We'd love to include it!

### Example Contribution Guidelines

1. **Clear Purpose:** Example should demonstrate a specific use case
2. **Well Documented:** Include comprehensive README with setup instructions
3. **Production Quality:** Use best practices and proper error handling
4. **Self-Contained:** Include all necessary files and sample data
5. **Tested:** Verify example works on clean installation

### Submission Process

1. Fork the repository
2. Create example in `examples/your-example-name/`
3. Follow the standard example structure
4. Test thoroughly
5. Submit pull request with description

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

---

## 📝 Example Template

Use this template when creating new examples:

````markdown
# Example Name

Brief description of what this example demonstrates.

## Overview

**What you'll learn:**
- Key concept 1
- Key concept 2
- Key concept 3

**Time to complete:** X minutes

**Difficulty:** ⭐ Beginner / ⭐⭐ Intermediate / ⭐⭐⭐ Advanced

## Prerequisites

- NeuronDB installed and running
- Component X installed (if applicable)
- API keys (if applicable)

## Setup

Step-by-step setup instructions...

## Usage

How to run the example...

## Explanation

Detailed explanation of key concepts...

## Next Steps

What to explore next...

## Troubleshooting

Common issues and solutions...
````

---

## 🆘 Support

### Getting Help

| Resource | Description |
|----------|-------------|
| **Example READMEs** | Each example has detailed documentation |
| **[Troubleshooting Guide](../NeuronDB/docs/troubleshooting.md)** | Common issues and solutions |
| **[GitHub Discussions](https://github.com/neurondb/NeurondB/discussions)** | Ask questions and share ideas |
| **Email Support** | support@neurondb.ai |

### Reporting Issues

If you find issues with an example:

1. Check the example's README for troubleshooting steps
2. Verify all prerequisites are met
3. Check logs for specific error messages
4. Report issues with:
   - Example name
   - Steps to reproduce
   - Error messages
   - Environment details

---

## 📊 Example Statistics

| Metric | Count |
|--------|-------|
| **Total Examples** | 4 main examples |
| **Additional Code Examples** | 100+ (across components) |
| **SQL Demos** | 60+ in NeuronDB/demo/ |
| **Python Examples** | 24 in NeuronAgent/examples/ |
| **Coverage** | All major use cases covered |

---

**Last Updated:** 2025-01-30  
**Examples Version:** 1.0.0

**Happy Building! 🚀**

