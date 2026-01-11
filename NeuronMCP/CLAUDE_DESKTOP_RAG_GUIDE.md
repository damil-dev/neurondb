# Using RAG with Claude Desktop → NeuronMCP → NeuronDB

This guide shows how to load sample data, generate embeddings, and test RAG using Claude Desktop with NeuronMCP.

## Prerequisites

✅ NeuronMCP server is running and connected (visible in Claude Desktop → Settings → Local MCP servers)

## Step-by-Step Guide

### Step 1: Load Sample Documents

Ask Claude Desktop to ingest sample documents:

**Prompt:**
```
Use the neurondb_ingest_documents tool to load the following document into a collection called "knowledge_base":

"Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's extensive standard library and third-party packages make it suitable for web development, data science, machine learning, and automation tasks."
```

**Additional Documents to Load:**
```
Use neurondb_ingest_documents to add this document to the "knowledge_base" collection:

"PostgreSQL is a powerful, open-source relational database management system. It supports advanced features like full-text search, JSON support, and extensibility through custom extensions. PostgreSQL is ACID-compliant and handles complex queries efficiently."
```

```
Use neurondb_ingest_documents to add this to "knowledge_base":

"NeuronDB is a vector database extension for PostgreSQL that enables efficient similarity search and machine learning operations on high-dimensional vectors. It supports HNSW and IVF indexes for fast approximate nearest neighbor search."
```

```
Use neurondb_ingest_documents to add this to "knowledge_base":

"Vector embeddings are numerical representations of text, images, or other data that capture semantic meaning. They enable similarity search and retrieval-augmented generation (RAG) by allowing systems to find semantically similar content."
```

### Step 2: Test Vector Search / Context Retrieval

**Prompt:**
```
Use the neurondb_retrieve_context tool to search the "knowledge_base" collection for information about Python programming. Retrieve the top 3 most relevant chunks.
```

### Step 3: Test RAG with Citations

**Prompt:**
```
Use the neurondb_answer_with_citations tool to answer this question using the "knowledge_base" collection:

"What is Python programming language and what are its key features?"
```

### Step 4: Verify Data Was Loaded

**Prompt:**
```
Check what data is in the knowledge_base collection. How many documents/chunks are stored?
```

## Available RAG Tools

### `neurondb_ingest_documents`
- **Purpose:** Ingest documents with automatic chunking and embedding
- **Parameters:**
  - `collection` (required): Table/collection name
  - `source` (required): Document text to ingest
  - `chunk_size` (optional, default: 500): Characters per chunk
  - `overlap` (optional, default: 50): Overlap between chunks
  - `embedding_model` (optional): Model name (uses default if not specified)

### `neurondb_retrieve_context`
- **Purpose:** Retrieve relevant context using vector search
- **Parameters:**
  - `query` (required): Search query text
  - `table` (required): Table/collection name
  - `vector_column` (required): Vector column name (usually "embedding")
  - `limit` (optional, default: 5): Number of results to return

### `neurondb_answer_with_citations`
- **Purpose:** Generate answer using RAG with source citations
- **Parameters:**
  - `collection` (required): Collection name to search
  - `query` (required): Question to answer
  - `k` (optional, default: 5): Number of context chunks to retrieve
  - `model` (optional): LLM model for answer generation

### `neurondb_chunk_document`
- **Purpose:** Chunk a document into smaller pieces
- **Parameters:**
  - `text` (required): Text to chunk
  - `chunk_size` (optional, default: 500): Characters per chunk
  - `overlap` (optional, default: 50): Overlap between chunks

### `neurondb_generate_embedding`
- **Purpose:** Generate text embedding
- **Parameters:**
  - `text` (required): Text to embed
  - `model` (optional): Model name (uses default if not specified)

## Example Complete RAG Workflow

Copy and paste this into Claude Desktop:

```
I want to set up a RAG system. Please:

1. Use neurondb_ingest_documents to create a collection called "docs" and load these documents:

Document 1: "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming. It includes supervised learning, unsupervised learning, and reinforcement learning."

Document 2: "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for image recognition, natural language processing, and speech recognition."

Document 3: "Vector databases store high-dimensional vectors and enable fast similarity search. They are essential for RAG systems, recommendation engines, and semantic search applications."

2. After loading, use neurondb_retrieve_context to search for information about "machine learning and neural networks" from the "docs" collection.

3. Finally, use neurondb_answer_with_citations to answer: "What is the difference between machine learning and deep learning?" using the "docs" collection.
```

## Troubleshooting

If Claude doesn't use the tools automatically:

1. **Be explicit:** Start with "Use the [tool_name] tool to..."
2. **List tools first:** Ask "What tools are available from NeuronMCP?" 
3. **Check connection:** Verify NeuronMCP shows as "running" in Claude Desktop settings

## Notes

- The `neurondb_ingest_documents` tool automatically:
  - Chunks the document
  - Generates embeddings for each chunk
  - Inserts chunks and embeddings into the collection table
  
- The collection table is created automatically if it doesn't exist (with `text` and `embedding` columns)

- All embeddings use the default embedding model configured in NeuronDB unless specified

