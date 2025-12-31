.PHONY: build run test migrate clean

build:
	go build -o bin/neurondesk-api ./cmd/server

run:
	go run ./cmd/server

test:
	@echo "Running tests with race detector..."
	go test -race ./...

test-fast:
	@echo "Running tests (fast mode, no race detector)..."
	go test ./...

migrate:
	psql -d neurondesk -f migrations/001_initial_schema.sql

clean:
	rm -rf bin/

