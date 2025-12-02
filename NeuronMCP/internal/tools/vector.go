package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// VectorSearchTool performs vector similarity search
type VectorSearchTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchTool creates a new vector search tool
func NewVectorSearchTool(db *database.Database, logger *logging.Logger) *VectorSearchTool {
	return &VectorSearchTool{
		BaseTool: NewBaseTool(
			"vector_search",
			"Perform vector similarity search using L2, cosine, inner product, L1, Hamming, Chebyshev, or Minkowski distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name containing vectors",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Name of the vector column",
					},
					"query_vector": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "number"},
						"description": "Query vector for similarity search",
					},
					"limit": map[string]interface{}{
						"type":        "number",
						"default":     10,
						"minimum":     1,
						"maximum":     1000,
						"description": "Maximum number of results",
					},
					"distance_metric": map[string]interface{}{
						"type":        "string",
						"enum":        []interface{}{"l2", "cosine", "inner_product", "l1", "hamming", "chebyshev", "minkowski"},
						"default":     "l2",
						"description": "Distance metric to use",
					},
					"additional_columns": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Additional columns to return in results",
					},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the vector search
func (t *VectorSearchTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for vector_search tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}
	distanceMetric := "l2"
	if dm, ok := params["distance_metric"].(string); ok {
		distanceMetric = dm
	}
	additionalColumns := []interface{}{}
	if ac, ok := params["additional_columns"].([]interface{}); ok {
		additionalColumns = ac
	}

	if table == "" {
		return Error("table parameter is required and cannot be empty for vector_search tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for vector_search tool on table '%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	if queryVector == nil || len(queryVector) == 0 {
		return Error(fmt.Sprintf("query_vector parameter is required and cannot be empty for vector_search tool on table '%s', column '%s'", table, vectorColumn), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":    "query_vector",
			"table":        table,
			"vector_column": vectorColumn,
			"params":       params,
		}), nil
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, distanceMetric, limit, additionalColumns)
	if err != nil {
		t.logger.Error("Vector search failed", err, params)
		return Error(fmt.Sprintf("Vector search execution failed: table='%s', vector_column='%s', distance_metric='%s', limit=%d, query_vector_dimension=%d, additional_columns_count=%d, error=%v", table, vectorColumn, distanceMetric, limit, len(queryVector), len(additionalColumns), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   distanceMetric,
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": distanceMetric,
		"table":          table,
		"vector_column":  vectorColumn,
		"limit":         limit,
	}), nil
}

// VectorSearchL2Tool performs L2 distance vector search
type VectorSearchL2Tool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchL2Tool creates a new L2 vector search tool
func NewVectorSearchL2Tool(db *database.Database, logger *logging.Logger) *VectorSearchL2Tool {
	return &VectorSearchL2Tool{
		BaseTool: NewBaseTool(
			"vector_search_l2",
			"Perform vector similarity search using L2 (Euclidean) distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the L2 vector search
func (t *VectorSearchL2Tool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "l2", limit, nil)
	if err != nil {
		t.logger.Error("L2 vector search failed", err, params)
		return Error(fmt.Sprintf("L2 vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "l2",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "l2",
	}), nil
}

// VectorSearchCosineTool performs cosine distance vector search
type VectorSearchCosineTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchCosineTool creates a new cosine vector search tool
func NewVectorSearchCosineTool(db *database.Database, logger *logging.Logger) *VectorSearchCosineTool {
	return &VectorSearchCosineTool{
		BaseTool: NewBaseTool(
			"vector_search_cosine",
			"Perform vector similarity search using cosine distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the cosine vector search
func (t *VectorSearchCosineTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "cosine", limit, nil)
	if err != nil {
		t.logger.Error("Cosine vector search failed", err, params)
		return Error(fmt.Sprintf("Cosine vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "cosine",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "cosine",
	}), nil
}

// VectorSearchInnerProductTool performs inner product distance vector search
type VectorSearchInnerProductTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewVectorSearchInnerProductTool creates a new inner product vector search tool
func NewVectorSearchInnerProductTool(db *database.Database, logger *logging.Logger) *VectorSearchInnerProductTool {
	return &VectorSearchInnerProductTool{
		BaseTool: NewBaseTool(
			"vector_search_inner_product",
			"Perform vector similarity search using inner product distance",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table":         map[string]interface{}{"type": "string"},
					"vector_column": map[string]interface{}{"type": "string"},
					"query_vector":  map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "number"}},
					"limit":         map[string]interface{}{"type": "number", "default": 10, "minimum": 1, "maximum": 1000},
				},
				"required": []interface{}{"table", "vector_column", "query_vector"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the inner product vector search
func (t *VectorSearchInnerProductTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	queryVector, _ := params["query_vector"].([]interface{})
	limit := 10
	if l, ok := params["limit"].(float64); ok {
		limit = int(l)
	}

	results, err := t.executor.ExecuteVectorSearch(ctx, table, vectorColumn, queryVector, "inner_product", limit, nil)
	if err != nil {
		t.logger.Error("Inner product vector search failed", err, params)
		return Error(fmt.Sprintf("Inner product vector search execution failed: table='%s', vector_column='%s', limit=%d, query_vector_dimension=%d, error=%v", table, vectorColumn, limit, len(queryVector), err), "SEARCH_ERROR", map[string]interface{}{
			"table":             table,
			"vector_column":     vectorColumn,
			"distance_metric":   "inner_product",
			"limit":            limit,
			"query_vector_size": len(queryVector),
			"error":            err.Error(),
		}), nil
	}

	return Success(results, map[string]interface{}{
		"count":          len(results),
		"distance_metric": "inner_product",
	}), nil
}

// GenerateEmbeddingTool generates text embeddings
type GenerateEmbeddingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewGenerateEmbeddingTool creates a new embedding generation tool
func NewGenerateEmbeddingTool(db *database.Database, logger *logging.Logger) *GenerateEmbeddingTool {
	return &GenerateEmbeddingTool{
		BaseTool: NewBaseTool(
			"generate_embedding",
			"Generate text embedding using configured model",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"text": map[string]interface{}{
						"type":        "string",
						"description": "Text to embed",
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Model name (optional, uses default if not specified)",
					},
				},
				"required": []interface{}{"text"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the embedding generation
func (t *GenerateEmbeddingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	text, _ := params["text"].(string)
	model, _ := params["model"].(string)
	
	textLen := len(text)
	if textLen == 0 {
		return Error("text parameter is required and cannot be empty for generate_embedding tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "text",
			"text_length": 0,
			"params":      params,
		}), nil
	}

	modelName := model
	if modelName == "" {
		modelName = "default"
	}

	// Use NeuronDB's unified embedding function: neurondb.embed(model, input_text, task)
	// or neurondb.generate_embedding(model, text) for backward compatibility
	query := "SELECT neurondb.embed($1, $2, 'embedding') AS embedding"
	queryParams := []interface{}{modelName, text}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Embedding generation failed", err, params)
		return Error(fmt.Sprintf("Embedding generation failed: text_length=%d, model='%s', error=%v", textLen, modelName, err), "EMBEDDING_ERROR", map[string]interface{}{
			"text_length": textLen,
			"model":       modelName,
			"error":       err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{"model": modelName}), nil
}

// BatchEmbeddingTool generates embeddings for multiple texts
type BatchEmbeddingTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewBatchEmbeddingTool creates a new batch embedding tool
func NewBatchEmbeddingTool(db *database.Database, logger *logging.Logger) *BatchEmbeddingTool {
	return &BatchEmbeddingTool{
		BaseTool: NewBaseTool(
			"batch_embedding",
			"Generate embeddings for multiple texts efficiently",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"texts": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Array of texts to embed",
						"minItems":    1,
						"maxItems":    1000,
					},
					"model": map[string]interface{}{
						"type":        "string",
						"description": "Model name (optional)",
					},
				},
				"required": []interface{}{"texts"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the batch embedding
func (t *BatchEmbeddingTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error("Invalid parameters", "VALIDATION_ERROR", map[string]interface{}{"errors": errors}), nil
	}

	texts, _ := params["texts"].([]interface{})
	model, _ := params["model"].(string)
	
	textsCount := len(texts)
	if textsCount == 0 {
		return Error("texts parameter is required and cannot be empty array for batch_embedding tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "texts",
			"texts_count": 0,
			"params":      params,
		}), nil
	}

	if textsCount > 1000 {
		return Error(fmt.Sprintf("texts array exceeds maximum size of 1000: received %d texts for batch_embedding tool", textsCount), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":   "texts",
			"texts_count": textsCount,
			"max_count":   1000,
			"params":      params,
		}), nil
	}

	modelName := model
	if modelName == "" {
		modelName = "default"
	}

	// Use NeuronDB's batch embedding function: neurondb.embed_batch(model, texts[])
	query := "SELECT neurondb.embed_batch($1, $2) AS embeddings"
	queryParams := []interface{}{modelName, texts}

	result, err := t.executor.ExecuteQueryOne(ctx, query, queryParams)
	if err != nil {
		t.logger.Error("Batch embedding failed", err, params)
		return Error(fmt.Sprintf("Batch embedding generation failed: texts_count=%d, model='%s', error=%v", textsCount, modelName, err), "EMBEDDING_ERROR", map[string]interface{}{
			"texts_count": textsCount,
			"model":       modelName,
			"error":       err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"count": len(texts),
		"model": modelName,
	}), nil
}

