.PHONY: build clean test run build-client

build:
	go build -o bin/neuronmcp ./cmd/neurondb-mcp

build-client:
	go build -o bin/neurondb-mcp-client ./cmd/neurondb-mcp-client

clean:
	rm -rf bin/

test:
	@echo "Running tests with race detector..."
	go test -race ./...

test-fast:
	@echo "Running tests (fast mode, no race detector)..."
	go test ./...

run: build
	./bin/neuronmcp

