/*-------------------------------------------------------------------------
 *
 * register.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/tools/register.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"github.com/neurondb/NeuronMCP/internal/database"
	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* RegisterAllTools registers all available tools with the registry */
func RegisterAllTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
  /* Vector search tools */
	registry.Register(NewVectorSearchTool(db, logger))
	registry.Register(NewVectorSearchL2Tool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))
	registry.Register(NewVectorSearchInnerProductTool(db, logger))
	registry.Register(NewVectorSearchL1Tool(db, logger))
	registry.Register(NewVectorSearchHammingTool(db, logger))
	registry.Register(NewVectorSearchChebyshevTool(db, logger))
	registry.Register(NewVectorSearchMinkowskiTool(db, logger))

  /* Embedding tools */
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewBatchEmbeddingTool(db, logger))

  /* Additional vector tools */
	registry.Register(NewVectorSimilarityTool(db, logger))
	registry.Register(NewCreateVectorIndexTool(db, logger))

  /* ML tools */
	registry.Register(NewTrainModelTool(db, logger))
	registry.Register(NewPredictTool(db, logger))
	registry.Register(NewEvaluateModelTool(db, logger))
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))
	registry.Register(NewDeleteModelTool(db, logger))

  /* Analytics tools */
	registry.Register(NewClusterDataTool(db, logger))
	registry.Register(NewDetectOutliersTool(db, logger))
	registry.Register(NewReduceDimensionalityTool(db, logger))

  /* RAG tools */
	registry.Register(NewProcessDocumentTool(db, logger))
	registry.Register(NewRetrieveContextTool(db, logger))
	registry.Register(NewGenerateResponseTool(db, logger))

  /* Composite RAG tools */
	registry.Register(NewIngestDocumentsTool(db, logger))
	registry.Register(NewAnswerWithCitationsTool(db, logger))
	registry.Register(NewChunkDocumentTool(db, logger))

  /* Indexing tools */
	registry.Register(NewCreateHNSWIndexTool(db, logger))
	registry.Register(NewCreateIVFIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))
	registry.Register(NewDropIndexTool(db, logger))
	registry.Register(NewTuneHNSWIndexTool(db, logger))
	registry.Register(NewTuneIVFIndexTool(db, logger))

  /* Additional ML tools */
	registry.Register(NewPredictBatchTool(db, logger))
	registry.Register(NewExportModelTool(db, logger))

  /* Analytics tools */
	registry.Register(NewAnalyzeDataTool(db, logger))

  /* Hybrid search tools */
	registry.Register(NewHybridSearchTool(db, logger))
	registry.Register(NewTextSearchTool(db, logger))
	registry.Register(NewReciprocalRankFusionTool(db, logger))
	registry.Register(NewSemanticKeywordSearchTool(db, logger))
	registry.Register(NewMultiVectorSearchTool(db, logger))
	registry.Register(NewFacetedVectorSearchTool(db, logger))
	registry.Register(NewTemporalVectorSearchTool(db, logger))
	registry.Register(NewDiverseVectorSearchTool(db, logger))

  /* Reranking tools */
	registry.Register(NewRerankCrossEncoderTool(db, logger))
	registry.Register(NewRerankLLMTool(db, logger))
	registry.Register(NewRerankCohereTool(db, logger))
	registry.Register(NewRerankColBERTTool(db, logger))
	registry.Register(NewRerankLTRTool(db, logger))
	registry.Register(NewRerankEnsembleTool(db, logger))

  /* Advanced vector operations */
	registry.Register(NewVectorArithmeticTool(db, logger))
	registry.Register(NewVectorDistanceTool(db, logger))
	registry.Register(NewVectorSimilarityUnifiedTool(db, logger))

  /* Quantization tools */
	registry.Register(NewVectorQuantizationTool(db, logger))
	registry.Register(NewQuantizationAnalysisTool(db, logger))

  /* Complete embedding tools */
	registry.Register(NewEmbedImageTool(db, logger))
	registry.Register(NewEmbedMultimodalTool(db, logger))
	registry.Register(NewEmbedCachedTool(db, logger))
	registry.Register(NewConfigureEmbeddingModelTool(db, logger))
	registry.Register(NewGetEmbeddingModelConfigTool(db, logger))
	registry.Register(NewListEmbeddingModelConfigsTool(db, logger))
	registry.Register(NewDeleteEmbeddingModelConfigTool(db, logger))

  /* Quality metrics, drift detection, topic discovery */
	registry.Register(NewQualityMetricsTool(db, logger))
	registry.Register(NewDriftDetectionTool(db, logger))
	registry.Register(NewTopicDiscoveryTool(db, logger))

  /* Time series, AutoML, ONNX */
	registry.Register(NewTimeSeriesTool(db, logger))
	registry.Register(NewAutoMLTool(db, logger))
	registry.Register(NewONNXTool(db, logger))

  /* Vector graph operations */
	registry.Register(NewVectorGraphTool(db, logger))

  /* Vecmap operations */
	registry.Register(NewVecmapOperationsTool(db, logger))

  /* Dataset loading */
	registry.Register(NewDatasetLoadingTool(db, logger))

  /* Workers and GPU */
	registry.Register(NewWorkerManagementTool(db, logger))
	registry.Register(NewGPUMonitoringTool(db, logger))

  /* PostgreSQL tools - Server Information (8 tools) */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLStatsTool(db, logger))
	registry.Register(NewPostgreSQLDatabaseListTool(db, logger))
	registry.Register(NewPostgreSQLConnectionsTool(db, logger))
	registry.Register(NewPostgreSQLLocksTool(db, logger))
	registry.Register(NewPostgreSQLReplicationTool(db, logger))
	registry.Register(NewPostgreSQLSettingsTool(db, logger))
	registry.Register(NewPostgreSQLExtensionsTool(db, logger))

  /* PostgreSQL tools - Database Object Management (8 tools) */
	registry.Register(NewPostgreSQLTablesTool(db, logger))
	registry.Register(NewPostgreSQLIndexesTool(db, logger))
	registry.Register(NewPostgreSQLSchemasTool(db, logger))
	registry.Register(NewPostgreSQLViewsTool(db, logger))
	registry.Register(NewPostgreSQLSequencesTool(db, logger))
	registry.Register(NewPostgreSQLFunctionsTool(db, logger))
	registry.Register(NewPostgreSQLTriggersTool(db, logger))
	registry.Register(NewPostgreSQLConstraintsTool(db, logger))

  /* PostgreSQL tools - User and Role Management (3 tools) */
	registry.Register(NewPostgreSQLUsersTool(db, logger))
	registry.Register(NewPostgreSQLRolesTool(db, logger))
	registry.Register(NewPostgreSQLPermissionsTool(db, logger))

  /* PostgreSQL tools - Performance and Statistics (4 tools) */
	registry.Register(NewPostgreSQLTableStatsTool(db, logger))
	registry.Register(NewPostgreSQLIndexStatsTool(db, logger))
	registry.Register(NewPostgreSQLActiveQueriesTool(db, logger))
	registry.Register(NewPostgreSQLWaitEventsTool(db, logger))

  /* PostgreSQL tools - Size and Storage (4 tools) */
	registry.Register(NewPostgreSQLTableSizeTool(db, logger))
	registry.Register(NewPostgreSQLIndexSizeTool(db, logger))
	registry.Register(NewPostgreSQLBloatTool(db, logger))
	registry.Register(NewPostgreSQLVacuumStatsTool(db, logger))

  /* PostgreSQL tools - Administration (16 tools) */
	registry.Register(NewPostgreSQLExplainTool(db, logger))
	registry.Register(NewPostgreSQLExplainAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLVacuumTool(db, logger))
	registry.Register(NewPostgreSQLAnalyzeTool(db, logger))
	registry.Register(NewPostgreSQLReindexTool(db, logger))
	registry.Register(NewPostgreSQLTransactionsTool(db, logger))
	registry.Register(NewPostgreSQLTerminateQueryTool(db, logger))
	registry.Register(NewPostgreSQLSetConfigTool(db, logger))
	registry.Register(NewPostgreSQLReloadConfigTool(db, logger))
	registry.Register(NewPostgreSQLSlowQueriesTool(db, logger))
	registry.Register(NewPostgreSQLCacheHitRatioTool(db, logger))
	registry.Register(NewPostgreSQLBufferStatsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionsTool(db, logger))
	registry.Register(NewPostgreSQLPartitionStatsTool(db, logger))
	registry.Register(NewPostgreSQLFDWServersTool(db, logger))
	registry.Register(NewPostgreSQLFDWTablesTool(db, logger))
	registry.Register(NewPostgreSQLLogicalReplicationSlotsTool(db, logger))

  /* PostgreSQL tools - Query Execution & Management (6 tools) */
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanTool(db, logger))
	registry.Register(NewPostgreSQLCancelQueryTool(db, logger))
	registry.Register(NewPostgreSQLKillQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryHistoryTool(db, logger))
	registry.Register(NewPostgreSQLQueryOptimizationTool(db, logger))

  /* PostgreSQL tools - Backup & Recovery (6 tools) */
	registry.Register(NewPostgreSQLBackupDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLRestoreDatabaseTool(db, logger))
	registry.Register(NewPostgreSQLBackupTableTool(db, logger))
	registry.Register(NewPostgreSQLListBackupsTool(db, logger))
	registry.Register(NewPostgreSQLVerifyBackupTool(db, logger))
	registry.Register(NewPostgreSQLBackupScheduleTool(db, logger))

  /* PostgreSQL tools - Schema Modification (7 tools) */
	registry.Register(NewPostgreSQLCreateTableTool(db, logger))
	registry.Register(NewPostgreSQLAlterTableTool(db, logger))
	registry.Register(NewPostgreSQLDropTableTool(db, logger))
	registry.Register(NewPostgreSQLCreateIndexTool(db, logger))
	registry.Register(NewPostgreSQLCreateViewTool(db, logger))
	registry.Register(NewPostgreSQLCreateFunctionTool(db, logger))
	registry.Register(NewPostgreSQLCreateTriggerTool(db, logger))

  /* PostgreSQL tools - High Availability (5 tools) */
	registry.Register(NewPostgreSQLReplicationLagTool(db, logger))
	registry.Register(NewPostgreSQLPromoteReplicaTool(db, logger))
	registry.Register(NewPostgreSQLSyncStatusTool(db, logger))
	registry.Register(NewPostgreSQLClusterTool(db, logger))
	registry.Register(NewPostgreSQLFailoverTool(db, logger))

  /* PostgreSQL tools - Security & Compliance (4 tools) */
	registry.Register(NewPostgreSQLAuditLogTool(db, logger))
	registry.Register(NewPostgreSQLSecurityScanTool(db, logger))
	registry.Register(NewPostgreSQLComplianceCheckTool(db, logger))
	registry.Register(NewPostgreSQLEncryptionStatusTool(db, logger))

  /* PostgreSQL tools - Maintenance Operations (1 tool) */
	registry.Register(NewPostgreSQLMaintenanceWindowTool(db, logger))

  /* Advanced Vector Operations (10 tools) */
	registry.Register(NewVectorAggregateTool(db, logger))
	registry.Register(NewVectorNormalizeBatchTool(db, logger))
	registry.Register(NewVectorSimilarityMatrixTool(db, logger))
	registry.Register(NewVectorBatchDistanceTool(db, logger))
	registry.Register(NewVectorIndexStatisticsTool(db, logger))
	registry.Register(NewVectorDimensionReductionTool(db, logger))
	registry.Register(NewVectorClusterAnalysisTool(db, logger))
	registry.Register(NewVectorAnomalyDetectionTool(db, logger))
	registry.Register(NewVectorQuantizationAdvancedTool(db, logger))
	registry.Register(NewVectorCacheManagementTool(db, logger))

  /* Advanced ML Features (8 tools) */
	registry.Register(NewMLModelVersioningTool(db, logger))
	registry.Register(NewMLModelABTestingTool(db, logger))
	registry.Register(NewMLModelExplainabilityTool(db, logger))
	registry.Register(NewMLModelMonitoringTool(db, logger))
	registry.Register(NewMLModelRollbackTool(db, logger))
	registry.Register(NewMLModelRetrainingTool(db, logger))
	registry.Register(NewMLEnsembleModelsTool(db, logger))
	registry.Register(NewMLModelExportFormatsTool(db, logger))

  /* Advanced Graph Operations (6 tools) */
	registry.Register(NewVectorGraphShortestPathTool(db, logger))
	registry.Register(NewVectorGraphCentralityTool(db, logger))
	registry.Register(NewVectorGraphAnalysisTool(db, logger))
	registry.Register(NewVectorGraphCommunityDetectionAdvancedTool(db, logger))
	registry.Register(NewVectorGraphClusteringTool(db, logger))
	registry.Register(NewVectorGraphVisualizationTool(db, logger))

  /* Multi-Modal Operations (5 tools) */
	registry.Register(NewMultimodalEmbedTool(db, logger))
	registry.Register(NewMultimodalSearchTool(db, logger))
	registry.Register(NewMultimodalRetrievalTool(db, logger))
	registry.Register(NewImageEmbedBatchTool(db, logger))
	registry.Register(NewAudioEmbedTool(db, logger))
}

/* RegisterMinimalTools registers only essential tools for testing (approximately 15-20 tools) */
func RegisterMinimalTools(registry *ToolRegistry, db *database.Database, logger *logging.Logger) {
	if logger != nil {
		logger.Info("Registering minimal tool set for testing", nil)
	}

  /* Essential PostgreSQL tools - Server Info (3 tools) */
	registry.Register(NewPostgreSQLVersionTool(db, logger))
	registry.Register(NewPostgreSQLDatabaseListTool(db, logger))
	registry.Register(NewPostgreSQLTablesTool(db, logger))

  /* Essential PostgreSQL tools - Query Execution (2 tools) */
	registry.Register(NewPostgreSQLExecuteQueryTool(db, logger))
	registry.Register(NewPostgreSQLQueryPlanTool(db, logger))

  /* Essential Vector tools - Core Operations (3 tools) */
	registry.Register(NewGenerateEmbeddingTool(db, logger))
	registry.Register(NewVectorSearchTool(db, logger))
	registry.Register(NewVectorSearchCosineTool(db, logger))

  /* Essential Vector tools - Index Management (2 tools) */
	registry.Register(NewCreateVectorIndexTool(db, logger))
	registry.Register(NewIndexStatusTool(db, logger))

  /* Essential RAG tools (3 tools) */
	registry.Register(NewRetrieveContextTool(db, logger))
	registry.Register(NewChunkDocumentTool(db, logger))
	registry.Register(NewIngestDocumentsTool(db, logger))

  /* Essential ML tools (2 tools) */
	registry.Register(NewListModelsTool(db, logger))
	registry.Register(NewGetModelInfoTool(db, logger))

	if logger != nil {
		logger.Info("Minimal tool set registered successfully", nil)
	}
}

