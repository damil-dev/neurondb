# NeuronDB Examples - Complete Collection

**Professional, working examples demonstrating NeuronDB capabilities**

This directory contains complete, practical examples showing how to build real-world applications with NeuronDB.

---

## 🚀 Quick Start for Beginners

**New to NeuronDB? Start here:**

1. **[Basics Examples](basics/)** - Simple, copy-paste friendly examples
   - `01_basic_vectors.py` - Vector basics
   - `02_simple_embeddings.py` - Text to embeddings
   - `03_similarity_search.py` - Similarity search
   - `04_basic_index.py` - Creating indexes
   - `05_simple_rag.py` - Simple RAG system

2. **[Features Examples](features/)** - Examples by feature
   - Vector types, distance metrics, ML algorithms, indexing

3. **[Modules Examples](modules/)** - Examples by module
   - NeuronDB, NeuronAgent, NeuronMCP

---

## 📚 Example Categories

### 1. 🔍 Semantic Search (`semantic-search-docs/`)
Complete working example for semantic search over document collections.

**What you'll learn:**
- Document ingestion and chunking
- Embedding generation with Sentence Transformers
- Vector storage in NeuronDB
- Semantic search using HNSW indexes
- Similarity scoring

**Files:**
- `semantic_search.py` - Complete working implementation
- `readme.md` - Detailed documentation

**Quick Start:**
```bash
cd semantic-search-docs
pip install psycopg2-binary sentence-transformers numpy

# Run interactive demo
python semantic_search.py demo

# Or ingest your own documents
python semantic_search.py ingest --input-dir /path/to/docs

# Search
python semantic_search.py search --query "your question"
```

---

### 2. 💬 RAG Chatbot (`rag-chatbot-pdfs/`)
Full RAG (Retrieval-Augmented Generation) chatbot over PDF documents.

**What you'll learn:**
- PDF text extraction
- Document chunking strategies
- Embedding generation and storage
- Context retrieval
- LLM integration (OpenAI/Anthropic)
- Interactive chat interface

**Files:**
- `rag_chatbot.py` - Complete RAG implementation
- `readme.md` - Usage guide

**Quick Start:**
```bash
cd rag-chatbot-pdfs
pip install psycopg2-binary sentence-transformers pypdf openai anthropic python-dotenv

# Set API key
export OPENAI_API_KEY=your_key

# Run demo
python rag_chatbot.py demo

# Ingest PDFs
python rag_chatbot.py ingest --input-dir /path/to/pdfs

# Interactive chat
python rag_chatbot.py chat

# Single query
python rag_chatbot.py query --query "your question"
```

---

### 3. 🤖 Agent Tools (`agent-tools/`)
NeuronAgent with multiple tools (SQL, HTTP, custom tools).

**What you'll learn:**
- Agent creation and configuration
- Tool registration (SQL, HTTP, custom)
- Tool chaining for complex tasks
- Agent state management
- Error handling

**Files:**
- `readme.md` - Setup and examples

**Coming Soon:**
Complete implementation with working agent examples.

---

### 4. 🔌 MCP Integration (`mcp-integration/`)
Model Context Protocol integration examples.

**What you'll learn:**
- Claude Desktop configuration
- MCP server setup
- Tool discovery and usage
- Custom MCP clients
- Integration patterns

**Files:**
- `claude_desktop_config.json` - Claude Desktop setup
- `test_mcp_connection.py` - Connection testing
- `readme.md` - Complete guide

**Quick Start:**
```bash
cd mcp-integration

# Configure Claude Desktop
# Edit ~/.config/Claude/claude_desktop_config.json

# Test connection
python test_mcp_connection.py

# List tools
python list_tools.py
```

---

### 5. 📊 Data Loading (`data_loading/`)
Load datasets from HuggingFace Hub into NeuronDB.

**What you'll learn:**
- HuggingFace Datasets integration
- Auto-schema detection
- Batch loading
- Embedding generation
- Index creation

**Files:**
- `load_huggingface_dataset.py` - Complete loader

**Quick Start:**
```bash
cd data_loading
pip install datasets sentence-transformers psycopg2-binary

python load_huggingface_dataset.py \
    --dataset ag_news \
    --split train \
    --limit 1000 \
    --auto-embed \
    --create-indexes
```

---

### 6. 🧠 LLM Training (`llm_training/`)
Train custom LLM models for PostgreSQL-specific tasks.

**What you'll learn:**
- Custom model training
- Model export to Ollama
- LLM server setup
- PostgreSQL integration

**Files:**
- `train_postgres_llm.py` - Training script
- `export_to_ollama.sh` - Ollama export
- `start_custom_llm_system.sh` - Server startup
- `stop_custom_llm_system.sh` - Server shutdown

**Quick Start:**
```bash
cd llm_training

# Train model
python train_postgres_llm.py

# Start server
./start_custom_llm_system.sh

# Stop server
./stop_custom_llm_system.sh
```

---

## 🚀 Getting Started

### Prerequisites

**Core Requirements:**
- PostgreSQL 16+ with NeuronDB extension
- Python 3.8+
- pip

**Install NeuronDB first:**
```bash
# Using Docker (recommended for examples)
cd ../
./scripts/ecosystem-setup.sh --mode docker --all

# Or native installation
./scripts/install.sh
```

### Common Dependencies

Most examples require:
```bash
pip install psycopg2-binary sentence-transformers numpy
```

For RAG examples, also install:
```bash
pip install pypdf openai anthropic python-dotenv
```

For data loading:
```bash
pip install datasets
```

### Environment Variables

Set these for database connection:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=neurondb
export DB_USER=postgres
export DB_PASSWORD=neurondb
```

For LLM integration:
```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

---

## 📖 Example Walkthroughs

### Quick Demo: Semantic Search (5 minutes)

```bash
# 1. Navigate to example
cd semantic-search-docs

# 2. Install dependencies
pip install psycopg2-binary sentence-transformers numpy

# 3. Run demo (creates sample docs, ingests, searches)
python semantic_search.py demo
```

Output:
```
=================================================================
  NeuronDB Semantic Search Demo
=================================================================

✓ Created 3 sample documents in sample_docs/

=================================================================
  Step 1: Ingesting Documents
=================================================================

Processing: machine_learning.md
  Created 3 chunks
✓ Ingested machine_learning.md

...

=================================================================
  Step 2: Semantic Search
=================================================================

Query: "What is machine learning?"
------------------------------------------------------------------

1. machine_learning.md (chunk 0)
   Similarity: 0.8542
   Machine learning is a subset of artificial intelligence...
```

---

### Quick Demo: RAG Chatbot (5 minutes)

```bash
# 1. Navigate to example
cd rag-chatbot-pdfs

# 2. Install dependencies
pip install psycopg2-binary sentence-transformers pypdf openai

# 3. Set API key
export OPENAI_API_KEY=your_key

# 4. Run demo
python rag_chatbot.py demo
```

Output:
```
=================================================================
  NeuronDB RAG Chatbot Demo
=================================================================

Query: "What is machine learning?"
------------------------------------------------------------------
Answer: Based on the provided context, machine learning is a branch 
of artificial intelligence that enables systems to learn and improve 
from experience without being explicitly programmed...

Sources:
  1. ml_basics.txt (similarity: 0.912)
  2. database_intro.txt (similarity: 0.654)
```

---

## 🎓 Learning Path

### Beginners
1. Start with **Semantic Search** example
   - Understand document ingestion
   - Learn about embeddings
   - Practice vector search

2. Try **Data Loading** example
   - Load real datasets
   - Experiment with different embeddings
   - Understand indexing

### Intermediate
3. Build **RAG Chatbot**
   - Combine search with LLM
   - Learn context retrieval
   - Build interactive interfaces

4. Explore **Agent Tools**
   - Agent architecture
   - Tool orchestration
   - Complex workflows

### Advanced
5. Implement **MCP Integration**
   - Protocol integration
   - Custom clients
   - Production deployments

6. Train **Custom LLMs**
   - Model training
   - Fine-tuning
   - Deployment

---

## 🔧 Customization Guide

### Modify Embedding Models

```python
# In any example, change:
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
EMBEDDING_DIM = 384

# To:
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better quality, 768 dimensions
EMBEDDING_DIM = 768

# Or use multilingual:
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768
```

### Adjust Chunking Strategy

```python
# Modify chunk size and overlap:
CHUNK_SIZE = 500      # Increase for more context per chunk
CHUNK_OVERLAP = 50    # Increase for better continuity

# Or implement custom chunking:
def custom_chunk(text):
    # Split by paragraphs
    paragraphs = text.split('\n\n')
    return [p for p in paragraphs if len(p) > 100]
```

### Change LLM Provider

```python
# Switch between providers:
chatbot = RAGChatbot(DB_CONFIG, llm_provider="openai")     # GPT-3.5/4
chatbot = RAGChatbot(DB_CONFIG, llm_provider="anthropic")  # Claude
```

---

## 📊 Performance Tips

### Optimize Vector Search

```sql
-- Tune HNSW parameters
CREATE INDEX documents_embedding_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Increase for better recall (16-32)
    ef_construction = 64 -- Increase for better index quality (64-128)
);

-- Set runtime parameters
SET hnsw.ef_search = 100;  -- Increase for better recall (40-200)
```

### Batch Processing

```python
# Process documents in batches
BATCH_SIZE = 100

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    # Process batch
    embeddings = model.encode(batch, batch_size=32)
    # Store in database
```

### Connection Pooling

```python
from psycopg2.pool import SimpleConnectionPool

pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    **DB_CONFIG
)

# Use connection from pool
conn = pool.getconn()
# ... do work ...
pool.putconn(conn)
```

---

## 🐛 Troubleshooting

### Common Issues

#### "Extension neurondb does not exist"
```bash
# Install NeuronDB extension
cd ../scripts
./install.sh
```

#### "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

#### "CUDA out of memory" (when generating embeddings)
```python
# Use CPU instead
model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')

# Or reduce batch size
embeddings = model.encode(texts, batch_size=8)
```

#### "Connection refused" (database)
```bash
# Check if database is running
./scripts/health-check.sh

# Check connection parameters
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;"
```

---

## 📁 Example Structure

```
examples/
├── readme.md                          # This file
│
├── semantic-search-docs/              # ✅ Complete working example
│   ├── semantic_search.py
│   └── readme.md
│
├── rag-chatbot-pdfs/                  # ✅ Complete working example  
│   ├── rag_chatbot.py
│   └── readme.md
│
├── agent-tools/                       # 📋 Documentation + samples
│   └── readme.md
│
├── mcp-integration/                   # 📋 Configuration examples
│   └── readme.md
│
├── data_loading/                      # ✅ Complete working example
│   ├── load_huggingface_dataset.py
│   └── readme.md (to be created)
│
└── llm_training/                      # ✅ Complete working example
    ├── train_postgres_llm.py
    ├── export_to_ollama.sh
    ├── start_custom_llm_system.sh
    ├── stop_custom_llm_system.sh
    └── readme.md (to be created)
```

---

## 🔗 Related Documentation

### Core Documentation
- **[Scripts README](../scripts/readme.md)** - Production scripts
- **[Main README](../readme.md)** - Project overview
- **[Quick Start](../QUICKSTART.md)** - Getting started

### Component Documentation
- **[NeuronDB](../NeuronDB/readme.md)** - Database extension
- **[NeuronAgent](../NeuronAgent/readme.md)** - Agent runtime
- **[NeuronMCP](../NeuronMCP/readme.md)** - MCP server
- **[NeuronDesktop](../NeuronDesktop/readme.md)** - Web interface

### Advanced Guides
- **[RAG Playbook](../NeuronDB/docs/rag/playbook.md)** - Complete RAG guidance
- **[Vector Search](../NeuronDB/docs/vector-search/)** - Search optimization
- **[Performance](../NeuronDB/docs/performance/)** - Performance tuning

---

## 🤝 Contributing Examples

Want to add your own example?

1. Create a new directory in `examples/`
2. Include a working Python script or application
3. Add a comprehensive readme.md
4. Include requirements.txt
5. Test thoroughly
6. Submit a pull request

**Example Template:**
```
my-example/
├── readme.md           # Explain what it does, prerequisites, usage
├── example.py          # Complete working code
├── requirements.txt    # Python dependencies
└── sample_data/        # Optional sample data
```

---

## 📈 Example Comparison

| Example | Complexity | Time | Prerequisites | Use Case |
|---------|-----------|------|---------------|----------|
| **Semantic Search** | ⭐⭐ | 5 min | Basic | Document search |
| **Data Loading** | ⭐ | 10 min | Basic | Dataset import |
| **RAG Chatbot** | ⭐⭐⭐ | 15 min | LLM API key | Q&A systems |
| **Agent Tools** | ⭐⭐⭐⭐ | 30 min | NeuronAgent | Autonomous agents |
| **MCP Integration** | ⭐⭐⭐ | 20 min | MCP client | Claude Desktop |
| **LLM Training** | ⭐⭐⭐⭐⭐ | 2 hours | GPU (optional) | Custom models |

---

## 💡 Tips & Best Practices

### 1. Start Simple
Begin with semantic search before moving to complex RAG systems.

### 2. Use Sample Data
All examples include sample data generation for quick testing.

### 3. Monitor Performance
Use `monitor-status.sh` to watch resource usage during experiments.

### 4. Iterate on Chunking
Chunking strategy significantly impacts search quality. Experiment!

### 5. Test Embeddings
Try different embedding models to find the best fit for your data.

### 6. Optimize Indexes
HNSW parameters can be tuned for your specific recall/speed needs.

### 7. Version Control
Track your modifications to example code for reproducibility.

---

## 🎯 Next Steps

After exploring these examples:

1. **Build Your Application**
   - Combine concepts from multiple examples
   - Adapt to your specific use case
   - Scale to production

2. **Join the Community**
   - Share your examples
   - Ask questions
   - Contribute improvements

3. **Deploy to Production**
   - Use `scripts/ecosystem-setup.sh` for deployment
   - Follow deployment guides
   - Monitor with `scripts/monitor-status.sh`

---

**Last Updated:** 2025-12-31  
**Examples Version:** 2.0.0  
**Status:** ✅ Maintained

---

## 📬 Support

Questions about examples?

- **GitHub Issues**: [github.com/neurondb/neurondb/issues](https://github.com/neurondb/neurondb/issues)
- **Documentation**: [www.neurondb.ai/docs](https://www.neurondb.ai/docs)
- **Community Discord**: [discord.gg/neurondb](https://discord.gg/neurondb)

---

**Happy Building! 🚀**
