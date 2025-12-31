/*-------------------------------------------------------------------------
 *
 * main.go
 *    Main entry point for NeuronAgent server
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronAgent/cmd/agent-server/main.go
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronAgent/internal/agent"
	"github.com/neurondb/NeuronAgent/internal/api"
	"github.com/neurondb/NeuronAgent/internal/auth"
	"github.com/neurondb/NeuronAgent/internal/config"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/jobs"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/session"
	"github.com/neurondb/NeuronAgent/internal/tools"
	"github.com/neurondb/NeuronAgent/pkg/neurondb"
)

var (
	version   = "dev"
	buildDate = "unknown"
	gitCommit = "unknown"
)

func main() {
	var (
		showVersion = flag.Bool("version", false, "Show version information")
		showVersionShort = flag.Bool("v", false, "Show version information (short)")
		configPath  = flag.String("c", "", "Path to configuration file")
		configPathLong = flag.String("config", "", "Path to configuration file")
		showHelp    = flag.Bool("help", false, "Show help message")
		showHelpShort = flag.Bool("h", false, "Show help message (short)")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "NeuronAgent Server - AI Agent server for NeuronDB\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s                    Start server with default configuration\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -c config.yaml     Start server with custom config file\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --version          Show version information\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --help             Show this help message\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nConfiguration:\n")
		fmt.Fprintf(os.Stderr, "  Configuration can be provided via:\n")
		fmt.Fprintf(os.Stderr, "  - Command line flag: -c or --config\n")
		fmt.Fprintf(os.Stderr, "  - Environment variable: CONFIG_PATH\n")
		fmt.Fprintf(os.Stderr, "  - Environment variables (see config package for details)\n")
	}
	flag.Parse()

	/* Handle version flag */
	if *showVersion || *showVersionShort {
		fmt.Printf("neuronagent version %s\n", version)
		fmt.Printf("Build date: %s\n", buildDate)
		fmt.Printf("Git commit: %s\n", gitCommit)
		os.Exit(0)
	}

	/* Handle help flag */
	if *showHelp || *showHelpShort {
		flag.Usage()
		os.Exit(0)
	}

  /* Load configuration */
	cfg := config.DefaultConfig()
	
	/* Determine config path - command line flag takes precedence over environment variable */
	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = *configPathLong
	}
	if cfgPath == "" {
		cfgPath = os.Getenv("CONFIG_PATH")
	}
	
	if cfgPath != "" {
		var err error
		cfg, err = config.LoadConfig(cfgPath)
		if err != nil {
			fmt.Printf("Failed to load config: %v, using defaults\n", err)
		}
	} else {
   /* Load from environment variables if no config file */
		config.LoadFromEnv(cfg)
	}

  /* Initialize logging */
	metrics.InitLogging(cfg.Logging.Level, cfg.Logging.Format)

  /* Connect to database */
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		cfg.Database.Host, cfg.Database.Port, cfg.Database.User, cfg.Database.Password, cfg.Database.Database)

	connMaxIdleTime := 10 * time.Minute
	if cfg.Database.ConnMaxIdleTime > 0 {
		connMaxIdleTime = cfg.Database.ConnMaxIdleTime
	}
	
	database, err := db.NewDB(connStr, db.PoolConfig{
		MaxOpenConns:    cfg.Database.MaxOpenConns,
		MaxIdleConns:    cfg.Database.MaxIdleConns,
		ConnMaxLifetime: cfg.Database.ConnMaxLifetime,
		ConnMaxIdleTime: connMaxIdleTime,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: Failed to connect to database: %v\n", err)
		fmt.Fprintf(os.Stderr, "Connection string: host=%s port=%d user=%s dbname=%s\n", 
			cfg.Database.Host, cfg.Database.Port, cfg.Database.User, cfg.Database.Database)
		os.Exit(1)
	}
	defer database.Close()

  /* Run migrations */
	migrationRunner, err := db.NewMigrationRunner(database.DB, "./migrations")
	if err == nil {
		if err := migrationRunner.Run(context.Background()); err != nil {
			fmt.Printf("Warning: Migration failed: %v\n", err)
		}
	}

  /* Initialize components */
	queries := db.NewQueries(database.DB)
	queries.SetConnInfoFunc(database.GetConnInfoString)
	
	/* Initialize NeuronDB client */
	neurondbClient := neurondb.NewClient(database.DB)
	embedClient := neurondbClient.Embedding
	
	/* Initialize tool registry with NeuronDB clients */
	toolRegistry := tools.NewRegistryWithNeuronDB(queries, database, neurondbClient)
	runtime := agent.NewRuntime(database, queries, toolRegistry, embedClient)

  /* Initialize session management */
	sessionCache := session.NewCache(5 * time.Minute)
 	_ = session.NewManager(queries, sessionCache) /* Session manager for future use */
	sessionCleanup := session.NewCleanupService(queries, 1*time.Hour, 24*time.Hour)
	sessionCleanup.Start()
	defer sessionCleanup.Stop()

  /* Initialize API */
	handlers := api.NewHandlers(queries, runtime)
	keyManager := auth.NewAPIKeyManager(queries)
	principalManager := auth.NewPrincipalManager(queries)
	rateLimiter := auth.NewRateLimiter()

  /* Setup router */
	router := mux.NewRouter()
	router.Use(api.RequestIDMiddleware)
	router.Use(api.CORSMiddleware)
	router.Use(api.LoggingMiddleware)
	router.Use(api.AuthMiddleware(keyManager, principalManager, rateLimiter))

  /* API routes */
	apiRouter := router.PathPrefix("/api/v1").Subrouter()
	apiRouter.HandleFunc("/agents", handlers.CreateAgent).Methods("POST")
	apiRouter.HandleFunc("/agents", handlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", handlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", handlers.UpdateAgent).Methods("PUT")
	apiRouter.HandleFunc("/agents/{id}", handlers.DeleteAgent).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{id}/clone", handlers.CloneAgent).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/plan", handlers.GeneratePlan).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/reflect", handlers.ReflectOnResponse).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/delegate", handlers.DelegateToAgent).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/metrics", handlers.GetAgentMetrics).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/costs", handlers.GetAgentCosts).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/versions", handlers.ListAgentVersions).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/versions", handlers.CreateAgentVersion).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/versions/{version}", handlers.GetAgentVersion).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/versions/{version}/activate", handlers.ActivateAgentVersion).Methods("PUT")
	apiRouter.HandleFunc("/agents/{id}/relationships", handlers.ListAgentRelationships).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/relationships", handlers.CreateAgentRelationship).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/relationships/{relationship_id}", handlers.DeleteAgentRelationship).Methods("DELETE")
	apiRouter.HandleFunc("/plans", handlers.ListPlans).Methods("GET")
	apiRouter.HandleFunc("/plans/{id}", handlers.GetPlan).Methods("GET")
	apiRouter.HandleFunc("/plans/{id}", handlers.UpdatePlanStatus).Methods("PUT")
	apiRouter.HandleFunc("/reflections", handlers.ListReflections).Methods("GET")
	apiRouter.HandleFunc("/reflections/{id}", handlers.GetReflection).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/memory", handlers.ListMemoryChunks).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/memory/search", handlers.SearchMemory).Methods("POST")
	apiRouter.HandleFunc("/memory/{chunk_id}", handlers.GetMemoryChunk).Methods("GET")
	apiRouter.HandleFunc("/memory/{chunk_id}", handlers.DeleteMemoryChunk).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{id}/budget", handlers.GetBudget).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/budget", handlers.SetBudget).Methods("POST")
	apiRouter.HandleFunc("/agents/{id}/budget", handlers.UpdateBudget).Methods("PUT")
	apiRouter.HandleFunc("/agents/batch", handlers.BatchCreateAgents).Methods("POST")
	apiRouter.HandleFunc("/agents/batch/delete", handlers.BatchDeleteAgents).Methods("POST")
	apiRouter.HandleFunc("/messages/batch/delete", handlers.BatchDeleteMessages).Methods("POST")
	apiRouter.HandleFunc("/tools/batch/delete", handlers.BatchDeleteTools).Methods("POST")
	apiRouter.HandleFunc("/webhooks", handlers.ListWebhooks).Methods("GET")
	apiRouter.HandleFunc("/webhooks", handlers.CreateWebhook).Methods("POST")
	apiRouter.HandleFunc("/webhooks/{id}", handlers.GetWebhook).Methods("GET")
	apiRouter.HandleFunc("/webhooks/{id}", handlers.UpdateWebhook).Methods("PUT")
	apiRouter.HandleFunc("/webhooks/{id}", handlers.DeleteWebhook).Methods("DELETE")
	apiRouter.HandleFunc("/webhooks/{id}/deliveries", handlers.ListWebhookDeliveries).Methods("GET")
	apiRouter.HandleFunc("/approval-requests", handlers.ListApprovalRequests).Methods("GET")
	apiRouter.HandleFunc("/approval-requests/{id}", handlers.GetApprovalRequest).Methods("GET")
	apiRouter.HandleFunc("/approval-requests/{id}/approve", handlers.ApproveRequest).Methods("POST")
	apiRouter.HandleFunc("/approval-requests/{id}/reject", handlers.RejectRequest).Methods("POST")
	apiRouter.HandleFunc("/feedback", handlers.SubmitFeedback).Methods("POST")
	apiRouter.HandleFunc("/feedback", handlers.ListFeedback).Methods("GET")
	apiRouter.HandleFunc("/feedback/stats", handlers.GetFeedbackStats).Methods("GET")
	apiRouter.HandleFunc("/sessions", handlers.CreateSession).Methods("POST")
	apiRouter.HandleFunc("/sessions/{id}", handlers.GetSession).Methods("GET")
	apiRouter.HandleFunc("/sessions/{id}", handlers.UpdateSession).Methods("PUT")
	apiRouter.HandleFunc("/sessions/{id}", handlers.DeleteSession).Methods("DELETE")
	apiRouter.HandleFunc("/agents/{agent_id}/sessions", handlers.ListSessions).Methods("GET")
	apiRouter.HandleFunc("/sessions/{session_id}/messages", handlers.SendMessage).Methods("POST")
	apiRouter.HandleFunc("/sessions/{session_id}/messages", handlers.GetMessages).Methods("GET")
	apiRouter.HandleFunc("/messages/{id}", handlers.GetMessage).Methods("GET")
	apiRouter.HandleFunc("/messages/{id}", handlers.UpdateMessage).Methods("PUT")
	apiRouter.HandleFunc("/messages/{id}", handlers.DeleteMessage).Methods("DELETE")
	apiRouter.HandleFunc("/tools", handlers.ListTools).Methods("GET")
	apiRouter.HandleFunc("/tools", handlers.CreateTool).Methods("POST")
	apiRouter.HandleFunc("/tools/{name}", handlers.GetTool).Methods("GET")
	apiRouter.HandleFunc("/tools/{name}", handlers.UpdateTool).Methods("PUT")
	apiRouter.HandleFunc("/tools/{name}", handlers.DeleteTool).Methods("DELETE")
	apiRouter.HandleFunc("/tools/{name}/analytics", handlers.GetToolAnalytics).Methods("GET")
	apiRouter.HandleFunc("/memory/{id}/summarize", handlers.SummarizeMemory).Methods("POST")
	apiRouter.HandleFunc("/analytics/overview", handlers.GetAnalyticsOverview).Methods("GET")
	apiRouter.HandleFunc("/ws", api.HandleWebSocket(runtime)).Methods("GET")

  /* Health check */
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if err := database.HealthCheck(r.Context()); err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}).Methods("GET")

  /* Metrics endpoint (no auth required) */
	router.Handle("/metrics", metrics.Handler()).Methods("GET")

  /* Start background workers */
	queue := jobs.NewQueue(queries)
	processor := jobs.NewProcessor(database)
	worker := jobs.NewWorker(queue, processor, 5)
	worker.Start()
	defer worker.Stop()

  /* Start job scheduler */
	scheduler := jobs.NewScheduler(queue)
	scheduler.Start()
	defer scheduler.Stop()

  /* Start server */
	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

  /* Graceful shutdown */
	go func() {
		fmt.Printf("Server starting on %s\n", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "FATAL: Server failed to start on %s: %v\n", addr, err)
			os.Exit(1)
		}
	}()

  /* Wait for interrupt signal */
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	fmt.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		fmt.Printf("Server forced to shutdown: %v\n", err)
	}

	fmt.Println("Server exited")
}

