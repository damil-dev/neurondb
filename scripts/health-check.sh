#!/bin/bash
# Comprehensive health check script for production deployments

set -euo pipefail

echo "🏥 NeuronDB Ecosystem Health Check"
echo "==================================="

# Configuration
NEURONDB_HOST="${NEURONDB_HOST:-localhost}"
NEURONDB_PORT="${NEURONDB_PORT:-5432}"
NEURONAGENT_URL="${NEURONAGENT_URL:-http://localhost:8080}"
NEURONDESKTOP_URL="${NEURONDESKTOP_URL:-http://localhost:8081}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Health check function
check_service() {
    local name=$1
    local url=$2
    
    echo -n "Checking $name... "
    
    if response=$(curl -s -w "\n%{http_code}" -o /dev/null "$url" 2>/dev/null); then
        status_code=$(echo "$response" | tail -n1)
        if [ "$status_code" = "200" ]; then
            echo -e "${GREEN}✓ Healthy${NC}"
            return 0
        else
            echo -e "${RED}✗ Unhealthy (HTTP $status_code)${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Unreachable${NC}"
        return 1
    fi
}

# Database health
echo ""
echo "📊 Database Health"
echo "-----------------"
if PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$NEURONDB_HOST" -p "$NEURONDB_PORT" -U postgres -d postgres -c "SELECT version();" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL: Connected${NC}"
    
    # Check connections
    connections=$(PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$NEURONDB_HOST" -p "$NEURONDB_PORT" -U postgres -d postgres -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs)
    echo "  Active connections: $connections"
    
    # Check NeuronDB extension
    if PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$NEURONDB_HOST" -p "$NEURONDB_PORT" -U postgres -d postgres -c "SELECT neurondb_version();" >/dev/null 2>&1; then
        version=$(PGPASSWORD="${POSTGRES_PASSWORD:-postgres}" psql -h "$NEURONDB_HOST" -p "$NEURONDB_PORT" -U postgres -d postgres -t -c "SELECT neurondb_version();" 2>/dev/null | xargs)
        echo -e "${GREEN}✓ NeuronDB Extension: $version${NC}"
    else
        echo -e "${YELLOW}⚠ NeuronDB Extension: Not loaded${NC}"
    fi
else
    echo -e "${RED}✗ PostgreSQL: Connection failed${NC}"
fi

# Service health
echo ""
echo "📊 Service Health"
echo "-----------------"
check_service "NeuronAgent" "$NEURONAGENT_URL/health"
check_service "NeuronDesktop API" "$NEURONDESKTOP_URL/health"

# Disk space
echo ""
echo "📊 Disk Space"
echo "-------------"
df -h / | tail -n1 | awk '{print "  Used: " $3 " / " $2 " (" $5 ")"}'

# Memory
echo ""
echo "📊 Memory Usage"
echo "---------------"
if command -v free >/dev/null 2>&1; then
    free -h | grep Mem | awk '{print "  Used: " $3 " / " $2}'
fi

# Docker containers (if applicable)
echo ""
echo "📊 Docker Containers"
echo "--------------------"
if command -v docker >/dev/null 2>&1; then
    containers=$(docker ps --format "{{.Names}}" | grep -E "neurondb|neuronagent|neurondesk" || true)
    if [ -n "$containers" ]; then
        echo "$containers" | while read -r container; do
            status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
            if [ "$status" = "running" ]; then
                echo -e "  ${GREEN}✓${NC} $container: $status"
            else
                echo -e "  ${RED}✗${NC} $container: $status"
            fi
        done
    else
        echo -e "${YELLOW}⚠ No NeuronDB containers found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Docker not available${NC}"
fi

echo ""
echo "==================================="
echo "Health check complete"
