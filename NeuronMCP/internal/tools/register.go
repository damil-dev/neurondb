package tools

import (
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

// RegisterAllTools registers all available tools with the registry
func RegisterAllTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	// Vector search tools
	registry.Register(NewVectorSearchTool(db, logger))
	registry.Register(NewVectorSearchL2Tool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))
	registry.Register(NewVectorSearchInnerProductTool(db, logger))

	// Embedding tools
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewBatchEmbeddingTool(db, logger))

	// Additional vector tools
	registry.Register(NewVectorSimilarityTool(db, logger))
	registry.Register(NewCreateVectorIndexTool(db, logger))

	// ML tools
	registry.Register(NewTrainModelTool(db, logger))
	registry.Register(NewPredictTool(db, logger))
	registry.Register(NewEvaluateModelTool(db, logger))
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))
	registry.Register(NewDeleteModelTool(db, logger))

	// Analytics tools
	registry.Register(NewClusterDataTool(db, logger))
	registry.Register(NewDetectOutliersTool(db, logger))
	registry.Register(NewReduceDimensionalityTool(db, logger))

	// RAG tools
	registry.Register(NewProcessDocumentTool(db, logger))
	registry.Register(NewRetrieveContextTool(db, logger))
	registry.Register(NewGenerateResponseTool(db, logger))
	registry.Register(NewChunkDocumentTool(db, logger))

	// Indexing tools
	registry.Register(NewCreateHNSWIndexTool(db, logger))
	registry.Register(NewCreateIVFIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))
	registry.Register(NewDropIndexTool(db, logger))
	registry.Register(NewTuneHNSWIndexTool(db, logger))
	registry.Register(NewTuneIVFIndexTool(db, logger))

	// Additional ML tools
	registry.Register(NewPredictBatchTool(db, logger))
	registry.Register(NewExportModelTool(db, logger))

	// Analytics tools
	registry.Register(NewAnalyzeDataTool(db, logger))

	// Hybrid search tools
	registry.Register(NewHybridSearchTool(db, logger))
}

