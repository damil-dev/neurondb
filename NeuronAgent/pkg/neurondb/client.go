/*-------------------------------------------------------------------------
 *
 * client.go
 *    Database operations
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/pkg/neurondb/client.go
 *
 *-------------------------------------------------------------------------
 */

package neurondb

import (
	"github.com/jmoiron/sqlx"
)

/* Client provides a unified interface to NeuronDB functions */
type Client struct {
	Embedding     *EmbeddingClient
	LLM           *LLMClient
	ML            *MLClient
	Vector        *VectorClient
	RAG           *RAGClient
	Analytics     *AnalyticsClient
	HybridSearch  *HybridSearchClient
	Reranking     *RerankingClient
}

/* NewClient creates a new NeuronDB client */
func NewClient(db *sqlx.DB) *Client {
	return &Client{
		Embedding:    NewEmbeddingClient(db),
		LLM:          NewLLMClient(db),
		ML:           NewMLClient(db),
		Vector:       NewVectorClient(db),
		RAG:          NewRAGClient(db),
		Analytics:    NewAnalyticsClient(db),
		HybridSearch: NewHybridSearchClient(db),
		Reranking:    NewRerankingClient(db),
	}
}

