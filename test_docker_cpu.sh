#!/bin/bash
# Test script for CPU Docker setup
set -e

echo "=========================================="
echo "Testing CPU Docker Configuration"
echo "=========================================="

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Docker Compose config validation
echo -e "\n${YELLOW}[TEST 1]${NC} Validating docker-compose.yml syntax..."
if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose.yml syntax is valid"
else
    echo -e "${RED}✗${NC} docker-compose.yml has syntax errors"
    exit 1
fi

# Test 2: Check CPU service environment variables
echo -e "\n${YELLOW}[TEST 2]${NC} Checking CPU service environment variables..."
CPU_ENV=$(docker-compose config | grep -A 50 "neurondb:" | grep -E "NEURONDB_COMPUTE_MODE|NEURONDB_GPU_BACKEND_TYPE" | head -2)

if echo "$CPU_ENV" | grep -q "NEURONDB_COMPUTE_MODE.*'0'"; then
    echo -e "${GREEN}✓${NC} NEURONDB_COMPUTE_MODE is set to 0 (CPU mode)"
else
    echo -e "${RED}✗${NC} NEURONDB_COMPUTE_MODE is not set correctly"
    echo "Found: $CPU_ENV"
    exit 1
fi

if echo "$CPU_ENV" | grep -q "NEURONDB_GPU_BACKEND_TYPE.*'0'"; then
    echo -e "${GREEN}✓${NC} NEURONDB_GPU_BACKEND_TYPE is set to 0 (CPU backend)"
else
    echo -e "${RED}✗${NC} NEURONDB_GPU_BACKEND_TYPE is not set correctly"
    echo "Found: $CPU_ENV"
    exit 1
fi

# Test 3: Verify Dockerfile exists and has GPU_BACKENDS=none
echo -e "\n${YELLOW}[TEST 3]${NC} Checking CPU Dockerfile for GPU_BACKENDS=none..."
if grep -q "GPU_BACKENDS=none" NeuronDB/docker/Dockerfile; then
    echo -e "${GREEN}✓${NC} CPU Dockerfile correctly sets GPU_BACKENDS=none"
else
    echo -e "${RED}✗${NC} CPU Dockerfile missing GPU_BACKENDS=none"
    exit 1
fi

# Test 4: Verify initialization script has validation
echo -e "\n${YELLOW}[TEST 4]${NC} Checking initialization script..."
if grep -q "NEURONDB_COMPUTE_MODE" NeuronDB/docker/docker-entrypoint-initdb.d/10_configure_neurondb.sh && \
   grep -q "NEURONDB_GPU_BACKEND_TYPE" NeuronDB/docker/docker-entrypoint-initdb.d/10_configure_neurondb.sh; then
    echo -e "${GREEN}✓${NC} Initialization script handles environment variables"
else
    echo -e "${RED}✗${NC} Initialization script missing environment variable handling"
    exit 1
fi

# Test 5: Check for deprecated runtime: nvidia (should not exist)
echo -e "\n${YELLOW}[TEST 5]${NC} Checking for deprecated runtime: nvidia..."
if grep -q "runtime: nvidia" docker-compose.yml; then
    echo -e "${RED}✗${NC} Found deprecated 'runtime: nvidia' in docker-compose.yml"
    exit 1
else
    echo -e "${GREEN}✓${NC} No deprecated 'runtime: nvidia' found"
fi

# Test 6: Verify GPU services use deploy.resources.reservations.devices
echo -e "\n${YELLOW}[TEST 6]${NC} Checking CUDA service uses deploy.resources.reservations.devices..."
CUDA_CONFIG=$(docker-compose config | sed -n '/neurondb-cuda:/,/^  [a-z]/p')
if echo "$CUDA_CONFIG" | grep -q "driver: nvidia" && \
   ! echo "$CUDA_CONFIG" | grep -q "^[[:space:]]*runtime: nvidia"; then
    echo -e "${GREEN}✓${NC} CUDA service uses deploy.resources.reservations.devices (not deprecated runtime)"
else
    echo -e "${RED}✗${NC} CUDA service configuration issue"
    echo "Checking for 'driver: nvidia': $(echo "$CUDA_CONFIG" | grep -c 'driver: nvidia' || echo '0')"
    echo "Checking for 'runtime: nvidia': $(echo "$CUDA_CONFIG" | grep -c 'runtime: nvidia' || echo '0')"
    exit 1
fi

# Test 7: Verify all services have init: true
echo -e "\n${YELLOW}[TEST 7]${NC} Checking all services have init: true..."
SERVICES=("neurondb" "neurondb-cuda" "neurondb-rocm" "neurondb-metal")
ALL_HAVE_INIT=true
for service in "${SERVICES[@]}"; do
    SERVICE_CONFIG=$(docker-compose config | sed -n "/^  ${service}:/,/^  [a-z]/p")
    if echo "$SERVICE_CONFIG" | grep -q "init: true"; then
        echo -e "${GREEN}✓${NC} ${service} has init: true"
    else
        echo -e "${RED}✗${NC} ${service} missing init: true"
        ALL_HAVE_INIT=false
    fi
done
if [ "$ALL_HAVE_INIT" = false ]; then
    exit 1
fi

# Test 8: Verify GPU services have correct environment variables
echo -e "\n${YELLOW}[TEST 8]${NC} Checking GPU services environment variables..."
CUDA_ENV=$(docker-compose config | grep -A 50 "neurondb-cuda:" | grep -E "NEURONDB_COMPUTE_MODE|NEURONDB_GPU_BACKEND_TYPE" | head -2)
if echo "$CUDA_ENV" | grep -q "NEURONDB_COMPUTE_MODE.*'1'" && \
   echo "$CUDA_ENV" | grep -q "NEURONDB_GPU_BACKEND_TYPE.*'1'"; then
    echo -e "${GREEN}✓${NC} CUDA service has correct environment variables"
else
    echo -e "${RED}✗${NC} CUDA service environment variables incorrect"
    echo "Found: $CUDA_ENV"
    exit 1
fi

ROCM_ENV=$(docker-compose config | grep -A 50 "neurondb-rocm:" | grep -E "NEURONDB_COMPUTE_MODE|NEURONDB_GPU_BACKEND_TYPE" | head -2)
if echo "$ROCM_ENV" | grep -q "NEURONDB_COMPUTE_MODE.*'1'" && \
   echo "$ROCM_ENV" | grep -q "NEURONDB_GPU_BACKEND_TYPE.*'2'"; then
    echo -e "${GREEN}✓${NC} ROCm service has correct environment variables"
else
    echo -e "${RED}✗${NC} ROCm service environment variables incorrect"
    echo "Found: $ROCM_ENV"
    exit 1
fi

METAL_ENV=$(docker-compose config | grep -A 50 "neurondb-metal:" | grep -E "NEURONDB_COMPUTE_MODE|NEURONDB_GPU_BACKEND_TYPE" | head -2)
if echo "$METAL_ENV" | grep -q "NEURONDB_COMPUTE_MODE.*'1'" && \
   echo "$METAL_ENV" | grep -q "NEURONDB_GPU_BACKEND_TYPE.*'3'"; then
    echo -e "${GREEN}✓${NC} Metal service has correct environment variables"
else
    echo -e "${RED}✗${NC} Metal service environment variables incorrect"
    echo "Found: $METAL_ENV"
    exit 1
fi

echo -e "\n=========================================="
echo -e "${GREEN}All CPU Docker tests passed!${NC}"
echo "=========================================="

