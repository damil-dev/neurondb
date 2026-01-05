# NeuronDesktop API Dockerfile
# Built from repo root (see root docker-compose.yml) so we can also build NeuronMCP.

# --- Build NeuronDesktop API ---
FROM golang:1.24-alpine AS desktop_builder

WORKDIR /src/NeuronDesktop/api
RUN apk add --no-cache git

COPY NeuronDesktop/api/go.mod NeuronDesktop/api/go.sum ./
RUN go mod download

COPY NeuronDesktop/api/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /out/neurondesktop ./cmd/server

# --- Build NeuronMCP binary (for NeuronDesktop to spawn via stdio) ---
FROM golang:1.24-alpine AS mcp_builder

WORKDIR /src/NeuronMCP
RUN apk add --no-cache git

COPY NeuronMCP/go.mod NeuronMCP/go.sum ./
RUN go mod download

COPY NeuronMCP/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /out/neurondb-mcp ./cmd/neurondb-mcp

# --- Runtime image ---
FROM alpine:latest

RUN apk --no-cache add ca-certificates tzdata

WORKDIR /root/

COPY --from=desktop_builder /out/neurondesktop ./neurondesktop
COPY --from=mcp_builder /out/neurondb-mcp /usr/local/bin/neurondb-mcp
RUN chmod +x /usr/local/bin/neurondb-mcp

COPY NeuronDesktop/api/neurondesktop.sql ./neurondesktop.sql

EXPOSE 8081
CMD ["./neurondesktop"]

