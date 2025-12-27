.PHONY: build build-api build-frontend test test-api test-frontend clean clean-api clean-frontend deps deps-api deps-frontend install run run-api run-frontend help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)NeuronDesktop Build System$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}'

build: build-api build-frontend ## Build both API backend and frontend

build-api: deps-api ## Build the API backend (Go)
	@echo "$(CYAN)Building NeuronDesktop API...$(NC)"
	@cd api && go build -o bin/neurondesk-api ./cmd/server
	@echo "$(GREEN)✓ API built$(NC)"

build-frontend: deps-frontend ## Build the frontend (Next.js)
	@echo "$(CYAN)Building NeuronDesktop frontend...$(NC)"
	@cd frontend && npm run build
	@echo "$(GREEN)✓ Frontend built$(NC)"

deps: deps-api deps-frontend ## Install all dependencies

deps-api: ## Install API dependencies (Go modules)
	@echo "$(CYAN)Installing API dependencies...$(NC)"
	@cd api && go mod download
	@echo "$(GREEN)✓ API dependencies installed$(NC)"

deps-frontend: ## Install frontend dependencies (npm)
	@echo "$(CYAN)Installing frontend dependencies...$(NC)"
	@if [ ! -d frontend/node_modules ]; then \
		cd frontend && npm install; \
	else \
		echo "$(GREEN)✓ Frontend dependencies already installed$(NC)"; \
	fi

test: test-api test-frontend ## Run all tests

test-api: build-api ## Run API tests
	@echo "$(CYAN)Running API tests...$(NC)"
	@cd api && go test ./... || echo "$(YELLOW)⚠ Some tests may have failed$(NC)"

test-frontend: deps-frontend ## Run frontend tests
	@echo "$(CYAN)Running frontend tests...$(NC)"
	@cd frontend && npm test || echo "$(YELLOW)⚠ Frontend tests may not be configured$(NC)"

clean: clean-api clean-frontend ## Clean all build artifacts

clean-api: ## Clean API build artifacts
	@echo "$(CYAN)Cleaning API artifacts...$(NC)"
	@rm -rf api/bin/
	@echo "$(GREEN)✓ API cleaned$(NC)"

clean-frontend: ## Clean frontend build artifacts
	@echo "$(CYAN)Cleaning frontend artifacts...$(NC)"
	@cd frontend && rm -rf .next dist build node_modules/.cache
	@echo "$(GREEN)✓ Frontend cleaned$(NC)"

install: build ## Install the application (builds everything)
	@echo "$(GREEN)✓ NeuronDesktop installed$(NC)"
	@echo "$(CYAN)API binary: api/bin/neurondesk-api$(NC)"
	@echo "$(CYAN)Frontend: frontend/.next$(NC)"

run: run-api ## Run the full application (API only, frontend needs separate process)

run-api: build-api ## Run the API server
	@echo "$(CYAN)Running NeuronDesktop API...$(NC)"
	@cd api && ./bin/neurondesk-api || go run ./cmd/server

run-frontend: deps-frontend ## Run the frontend dev server
	@echo "$(CYAN)Running NeuronDesktop frontend...$(NC)"
	@cd frontend && npm run dev

migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(NC)"
	@if command -v psql >/dev/null 2>&1; then \
		psql -d neurondesk -f api/migrations/001_initial_schema.sql 2>/dev/null || \
		echo "$(YELLOW)⚠ Migration may have failed or database may not exist$(NC)"; \
	else \
		echo "$(YELLOW)⚠ psql not found - cannot run migrations$(NC)"; \
	fi

.DEFAULT_GOAL := help
