.PHONY: build test clean run migrate docker-build docker-up docker-down

# Build the application
build:
	@echo "Building NeuronAgent..."
	@go build -o bin/neuronagent cmd/agent-server/main.go

# Run tests with race detector (default - recommended for all testing)
test:
	@echo "Running tests with race detector..."
	@go test -v -race ./...

# Run tests without race detector (faster, less thorough)
test-fast:
	@echo "Running tests (fast mode, no race detector)..."
	@go test -v ./...

# Run tests with coverage (includes race detector)
test-coverage:
	@echo "Running tests with coverage and race detector..."
	@go test -v -race -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/ dist/ coverage.out coverage.html

# Run the application
run: build
	@echo "Running NeuronAgent..."
	@./bin/neuronagent

# Run database migrations
migrate:
	@echo "Running migrations..."
	@./scripts/neuronagent_migrate.sh

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...

# Lint code
lint:
	@echo "Linting code..."
	@golangci-lint run

# Docker build
docker-build:
	@echo "Building Docker image..."
	@docker build -t neuronagent:latest -f docker/Dockerfile .

# Docker compose up
docker-up:
	@echo "Starting Docker Compose..."
	@docker-compose -f docker/docker-compose.yml up -d

# Docker compose down
docker-down:
	@echo "Stopping Docker Compose..."
	@docker-compose -f docker/docker-compose.yml down

# Install dependencies
deps:
	@echo "Downloading dependencies..."
	@go mod download
	@go mod tidy

# Generate API keys
generate-key:
	@echo "Generating API key..."
	@./scripts/neuronagent_generate_keys.sh

