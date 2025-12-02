package tools

import (
	"context"
	"fmt"

	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// CreateHNSWIndexTool creates an HNSW index
type CreateHNSWIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewCreateHNSWIndexTool creates a new create HNSW index tool
func NewCreateHNSWIndexTool(db *database.Database, logger *logging.Logger) *CreateHNSWIndexTool {
	return &CreateHNSWIndexTool{
		BaseTool: NewBaseTool(
			"create_hnsw_index",
			"Create HNSW index for vector column",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name for the index",
					},
					"m": map[string]interface{}{
						"type":        "number",
						"default":     16,
						"minimum":     2,
						"maximum":     128,
						"description": "HNSW parameter M (connectivity)",
					},
					"ef_construction": map[string]interface{}{
						"type":        "number",
						"default":     200,
						"minimum":     4,
						"maximum":     2000,
						"description": "HNSW parameter ef_construction",
					},
				},
				"required": []interface{}{"table", "vector_column", "index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the HNSW index creation
func (t *CreateHNSWIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for create_hnsw_index tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	indexName, _ := params["index_name"].(string)
	m := 16
	if mVal, ok := params["m"].(float64); ok {
		m = int(mVal)
	}
	efConstruction := 200
	if ef, ok := params["ef_construction"].(float64); ok {
		efConstruction = int(ef)
	}

	if table == "" {
		return Error("table parameter is required and cannot be empty for create_hnsw_index tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for create_hnsw_index tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	if indexName == "" {
		return Error(fmt.Sprintf("index_name parameter is required and cannot be empty for create_hnsw_index tool: table='%s', vector_column='%s'", table, vectorColumn), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "index_name",
			"table":         table,
			"vector_column": vectorColumn,
			"params":        params,
		}), nil
	}

	if m < 2 || m > 128 {
		return Error(fmt.Sprintf("m parameter must be between 2 and 128 for create_hnsw_index tool: table='%s', vector_column='%s', index_name='%s', received m=%d", table, vectorColumn, indexName, m), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "m",
			"table":         table,
			"vector_column": vectorColumn,
			"index_name":    indexName,
			"m":             m,
			"min":           2,
			"max":           128,
			"params":        params,
		}), nil
	}

	if efConstruction < 4 || efConstruction > 2000 {
		return Error(fmt.Sprintf("ef_construction parameter must be between 4 and 2000 for create_hnsw_index tool: table='%s', vector_column='%s', index_name='%s', m=%d, received ef_construction=%d", table, vectorColumn, indexName, m, efConstruction), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":      "ef_construction",
			"table":          table,
			"vector_column":  vectorColumn,
			"index_name":     indexName,
			"m":              m,
			"ef_construction": efConstruction,
			"min":            4,
			"max":            2000,
			"params":         params,
		}), nil
	}

	// Use NeuronDB's unified index creation function
	// neurondb.create_index(table_name, vector_col, index_type, params)
	paramsJSON := fmt.Sprintf(`{"m": %d, "ef_construction": %d}`, m, efConstruction)
	query := `SELECT neurondb.create_index($1, $2, $3, $4::jsonb) AS result`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{
		table, vectorColumn, "hnsw", paramsJSON,
	})
	if err != nil {
		t.logger.Error("HNSW index creation failed", err, params)
		return Error(fmt.Sprintf("HNSW index creation execution failed: table='%s', vector_column='%s', index_name='%s', m=%d, ef_construction=%d, error=%v", table, vectorColumn, indexName, m, efConstruction, err), "INDEX_ERROR", map[string]interface{}{
			"table":           table,
			"vector_column":   vectorColumn,
			"index_name":      indexName,
			"m":               m,
			"ef_construction": efConstruction,
			"error":           err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"index_name":     indexName,
		"m":              m,
		"ef_construction": efConstruction,
	}), nil
}

// CreateIVFIndexTool creates an IVF index
type CreateIVFIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewCreateIVFIndexTool creates a new create IVF index tool
func NewCreateIVFIndexTool(db *database.Database, logger *logging.Logger) *CreateIVFIndexTool {
	return &CreateIVFIndexTool{
		BaseTool: NewBaseTool(
			"create_ivf_index",
			"Create IVF index for vector column",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"table": map[string]interface{}{
						"type":        "string",
						"description": "Table name",
					},
					"vector_column": map[string]interface{}{
						"type":        "string",
						"description": "Vector column name",
					},
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name for the index",
					},
					"num_lists": map[string]interface{}{
						"type":        "number",
						"default":     100,
						"minimum":     1,
						"description": "Number of lists for IVF",
					},
				},
				"required": []interface{}{"table", "vector_column", "index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the IVF index creation
func (t *CreateIVFIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for create_ivf_index tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	table, _ := params["table"].(string)
	vectorColumn, _ := params["vector_column"].(string)
	indexName, _ := params["index_name"].(string)
	numLists := 100
	if n, ok := params["num_lists"].(float64); ok {
		numLists = int(n)
	}

	if table == "" {
		return Error("table parameter is required and cannot be empty for create_ivf_index tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "table",
			"params":    params,
		}), nil
	}

	if vectorColumn == "" {
		return Error(fmt.Sprintf("vector_column parameter is required and cannot be empty for create_ivf_index tool: table='%s'", table), "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "vector_column",
			"table":     table,
			"params":    params,
		}), nil
	}

	if indexName == "" {
		return Error(fmt.Sprintf("index_name parameter is required and cannot be empty for create_ivf_index tool: table='%s', vector_column='%s'", table, vectorColumn), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "index_name",
			"table":         table,
			"vector_column": vectorColumn,
			"params":        params,
		}), nil
	}

	if numLists < 1 {
		return Error(fmt.Sprintf("num_lists must be at least 1 for create_ivf_index tool: table='%s', vector_column='%s', index_name='%s', received num_lists=%d", table, vectorColumn, indexName, numLists), "VALIDATION_ERROR", map[string]interface{}{
			"parameter":     "num_lists",
			"table":         table,
			"vector_column": vectorColumn,
			"index_name":    indexName,
			"num_lists":     numLists,
			"min":           1,
			"params":        params,
		}), nil
	}

	// Use NeuronDB's unified index creation function
	// neurondb.create_index(table_name, vector_col, index_type, params)
	paramsJSON := fmt.Sprintf(`{"num_lists": %d}`, numLists)
	query := `SELECT neurondb.create_index($1, $2, $3, $4::jsonb) AS result`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{
		table, vectorColumn, "ivf", paramsJSON,
	})
	if err != nil {
		t.logger.Error("IVF index creation failed", err, params)
		return Error(fmt.Sprintf("IVF index creation execution failed: table='%s', vector_column='%s', index_name='%s', num_lists=%d, error=%v", table, vectorColumn, indexName, numLists, err), "INDEX_ERROR", map[string]interface{}{
			"table":          table,
			"vector_column":  vectorColumn,
			"index_name":     indexName,
			"num_lists":      numLists,
			"error":          err.Error(),
		}), nil
	}

	return Success(result, map[string]interface{}{
		"index_name": indexName,
		"num_lists":  numLists,
	}), nil
}

// IndexStatusTool gets index status and statistics
type IndexStatusTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewIndexStatusTool creates a new index status tool
func NewIndexStatusTool(db *database.Database, logger *logging.Logger) *IndexStatusTool {
	return &IndexStatusTool{
		BaseTool: NewBaseTool(
			"index_status",
			"Get status and statistics for a vector index",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the index",
					},
				},
				"required": []interface{}{"index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the index status query
func (t *IndexStatusTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for index_status tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	indexName, _ := params["index_name"].(string)

	if indexName == "" {
		return Error("index_name parameter is required and cannot be empty for index_status tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "index_name",
			"params":    params,
		}), nil
	}

	query := `
		SELECT 
			schemaname,
			tablename,
			indexname,
			indexdef,
			pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
		FROM pg_indexes
		WHERE indexname = $1
	`
	result, err := t.executor.ExecuteQueryOne(ctx, query, []interface{}{indexName})
	if err != nil {
		t.logger.Error("Index status query failed", err, params)
		return Error(fmt.Sprintf("Index status query execution failed: index_name='%s', query='SELECT ... FROM pg_indexes WHERE indexname = $1', error=%v", indexName, err), "QUERY_ERROR", map[string]interface{}{
			"index_name": indexName,
			"query":      "SELECT ... FROM pg_indexes WHERE indexname = $1",
			"error":      err.Error(),
		}), nil
	}

	if result == nil || len(result) == 0 {
		return Error(fmt.Sprintf("Index not found in pg_indexes catalog: index_name='%s' (index may not exist or may not be accessible)", indexName), "NOT_FOUND", map[string]interface{}{
			"index_name": indexName,
			"catalog":    "pg_indexes",
		}), nil
	}

	return Success(result, map[string]interface{}{
		"index_name": indexName,
	}), nil
}

// DropIndexTool drops a vector index
type DropIndexTool struct {
	*BaseTool
	executor *QueryExecutor
	logger   *logging.Logger
}

// NewDropIndexTool creates a new drop index tool
func NewDropIndexTool(db *database.Database, logger *logging.Logger) *DropIndexTool {
	return &DropIndexTool{
		BaseTool: NewBaseTool(
			"drop_index",
			"Drop a vector index",
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"index_name": map[string]interface{}{
						"type":        "string",
						"description": "Name of the index to drop",
					},
				},
				"required": []interface{}{"index_name"},
			},
		),
		executor: NewQueryExecutor(db),
		logger:   logger,
	}
}

// Execute executes the index drop
func (t *DropIndexTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
	valid, errors := t.ValidateParams(params, t.InputSchema())
	if !valid {
		return Error(fmt.Sprintf("Invalid parameters for drop_index tool: %v", errors), "VALIDATION_ERROR", map[string]interface{}{
			"errors": errors,
			"params": params,
		}), nil
	}

	indexName, _ := params["index_name"].(string)

	if indexName == "" {
		return Error("index_name parameter is required and cannot be empty for drop_index tool", "VALIDATION_ERROR", map[string]interface{}{
			"parameter": "index_name",
			"params":    params,
		}), nil
	}

	// Escape identifier for safety
	escapedName := database.EscapeIdentifier(indexName)
	query := fmt.Sprintf("DROP INDEX IF EXISTS %s", escapedName)

	err := t.executor.Exec(ctx, query, nil)
	if err != nil {
		t.logger.Error("Index drop failed", err, params)
		return Error(fmt.Sprintf("Index drop execution failed: index_name='%s', escaped_name='%s', query='%s', error=%v", indexName, escapedName, query, err), "INDEX_ERROR", map[string]interface{}{
			"index_name":   indexName,
			"escaped_name": escapedName,
			"query":        query,
			"error":        err.Error(),
		}), nil
	}

	return Success(map[string]interface{}{
		"dropped":    true,
		"index_name": indexName,
	}, nil), nil
}

