/*-------------------------------------------------------------------------
 *
 * test_data.sql
 *    Test data setup for NeuronMCP tests
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/test/fixtures/test_data.sql
 *
 *-------------------------------------------------------------------------
 */

/* Create test schema */
CREATE SCHEMA IF NOT EXISTS test_schema;

/* Create test table with vector column */
CREATE TABLE IF NOT EXISTS test_schema.test_vectors (
    id SERIAL PRIMARY KEY,
    text_content TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

/* Insert sample vectors */
INSERT INTO test_schema.test_vectors (text_content, embedding, metadata) VALUES
    ('Sample text 1', '[0.1,0.2,0.3,0.4,0.5]'::vector, '{"category": "test"}'::jsonb),
    ('Sample text 2', '[0.2,0.3,0.4,0.5,0.6]'::vector, '{"category": "test"}'::jsonb),
    ('Sample text 3', '[0.3,0.4,0.5,0.6,0.7]'::vector, '{"category": "test"}'::jsonb)
ON CONFLICT DO NOTHING;

/* Create test table for ML training */
CREATE TABLE IF NOT EXISTS test_schema.test_ml_data (
    id SERIAL PRIMARY KEY,
    features vector(10),
    label FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

/* Insert sample ML training data */
INSERT INTO test_schema.test_ml_data (features, label) VALUES
    ('[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]'::vector, 1.0),
    ('[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1]'::vector, 2.0),
    ('[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2]'::vector, 3.0)
ON CONFLICT DO NOTHING;

/* Create test table for documents */
CREATE TABLE IF NOT EXISTS test_schema.test_documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

/* Insert sample documents */
INSERT INTO test_schema.test_documents (title, content, embedding, chunk_index) VALUES
    ('Test Document 1', 'This is a test document for RAG testing.', '[0.1,0.2,0.3,0.4,0.5]'::vector, 0),
    ('Test Document 2', 'Another test document for RAG testing.', '[0.2,0.3,0.4,0.5,0.6]'::vector, 0),
    ('Test Document 3', 'Yet another test document for RAG testing.', '[0.3,0.4,0.5,0.6,0.7]'::vector, 0)
ON CONFLICT DO NOTHING;

/* Create test index */
CREATE INDEX IF NOT EXISTS test_vectors_embedding_idx ON test_schema.test_vectors 
USING hnsw (embedding vector_cosine_ops);



