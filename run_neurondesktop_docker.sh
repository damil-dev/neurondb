#!/bin/bash
# Script to run NeuronDesktop in Docker
# This script provides an easy way to start the enhanced NeuronDesktop

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 NeuronDesktop Docker Launcher${NC}"
echo -e "${BLUE}================================${NC}\n"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo -e "${YELLOW}Please start Docker and try again${NC}"
    exit 1
fi

# Check if docker compose is available
if ! docker compose version > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: docker compose not found${NC}"
    echo -e "${YELLOW}Please install Docker Compose v2${NC}"
    exit 1
fi

# Change to NeuronDesktop directory
cd "$(dirname "$0")"

echo -e "${GREEN}📋 Available Options:${NC}\n"
echo -e "  ${BLUE}1.${NC} Start NeuronDesktop only (standalone)"
echo -e "  ${BLUE}2.${NC} Start full ecosystem (NeuronDB + NeuronAgent + NeuronMCP + NeuronDesktop)"
echo -e "  ${BLUE}3.${NC} Stop all containers"
echo -e "  ${BLUE}4.${NC} View logs"
echo -e "  ${BLUE}5.${NC} Rebuild and start"
echo ""

# Parse command line argument or prompt
if [ -n "$1" ]; then
    CHOICE="$1"
else
    read -p "Enter your choice (1-5): " CHOICE
fi

case $CHOICE in
    1)
        echo -e "\n${GREEN}🚀 Starting NeuronDesktop (standalone)...${NC}\n"
        cd "$(dirname "$0")/dockers/neurondesktop"
        docker compose up -d
        ;;
    2)
        echo -e "\n${GREEN}🚀 Starting full NeuronDB ecosystem...${NC}\n"
        docker compose --profile default up -d
        ;;
    3)
        echo -e "\n${YELLOW}🛑 Stopping all containers...${NC}\n"
        # Stop standalone neurondesktop
        if [ -f "$(dirname "$0")/dockers/neurondesktop/docker-compose.yml" ]; then
            cd "$(dirname "$0")/dockers/neurondesktop"
            docker compose down
        fi
        # Stop full ecosystem
        cd "$(dirname "$0")"
        docker compose --profile default down
        echo -e "${GREEN}✅ All containers stopped${NC}"
        exit 0
        ;;
    4)
        echo -e "\n${BLUE}📋 Viewing logs (Ctrl+C to exit)...${NC}\n"
        docker compose --profile default logs -f neurondesk-api neurondesk-frontend neurondb neuronagent
        exit 0
        ;;
    5)
        echo -e "\n${YELLOW}🔨 Rebuilding and starting...${NC}\n"
        docker compose --profile default build --no-cache neurondesk-api neurondesk-frontend
        docker compose --profile default up -d
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

# Wait for services to be healthy
echo -e "\n${YELLOW}⏳ Waiting for services to be healthy...${NC}\n"
sleep 5

# Check status
echo -e "\n${GREEN}✅ Service Status:${NC}\n"
docker compose --profile default ps

# Show access URLs
echo -e "\n${GREEN}🌐 Access URLs:${NC}"
echo -e "  ${BLUE}NeuronDesktop UI:${NC}    http://localhost:3000"
echo -e "  ${BLUE}NeuronDesktop API:${NC}   http://localhost:8081"
echo -e "  ${BLUE}API Health Check:${NC}    http://localhost:8081/health"
echo -e "  ${BLUE}API Metrics:${NC}         http://localhost:8081/metrics"
echo ""
echo -e "${GREEN}📚 Quick Start:${NC}"
echo -e "  1. Open ${BLUE}http://localhost:3000${NC} in your browser"
echo -e "  2. Login with default credentials (see documentation)"
echo -e "  3. Explore the unified dashboard"
echo ""
echo -e "${YELLOW}💡 Useful Commands:${NC}"
echo -e "  View logs:    ${BLUE}docker compose --profile default logs -f${NC}"
echo -e "  Stop all:     ${BLUE}./run_neurondesktop_docker.sh 3${NC}"
echo -e "  Restart:      ${BLUE}docker compose --profile default restart${NC}"
echo ""
echo -e "${GREEN}✨ NeuronDesktop is ready!${NC}\n"

