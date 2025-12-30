#!/bin/bash
# Deep scan of NeuronDB container libraries and dependencies

set -e

CONTAINER="${1:-neurondb-cpu}"

echo "=========================================="
echo "NeuronDB Container Library Deep Scan"
echo "Container: $CONTAINER"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if container is running
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
    echo -e "${RED}Error: Container $CONTAINER is not running${NC}"
    exit 1
fi

echo -e "${BLUE}1. PostgreSQL and Extension Status${NC}"
echo "----------------------------------------"
docker exec "$CONTAINER" psql -U neurondb -d neurondb -tAc "SELECT version();" 2>/dev/null | head -1
echo ""

# Check extension
EXT_STATUS=$(docker exec "$CONTAINER" psql -U neurondb -d neurondb -tAc "SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';" 2>/dev/null || echo "not installed")
if echo "$EXT_STATUS" | grep -q "neurondb"; then
    echo -e "${GREEN}✓ NeuronDB extension: $EXT_STATUS${NC}"
else
    echo -e "${YELLOW}⚠ NeuronDB extension not installed${NC}"
fi
echo ""

echo -e "${BLUE}2. Required System Libraries${NC}"
echo "----------------------------------------"

# Check for libcurl
if docker exec "$CONTAINER" find /usr/lib -name "*curl*" -o -name "libcurl*" 2>/dev/null | grep -q .; then
    CURL_LIBS=$(docker exec "$CONTAINER" find /usr/lib -name "*curl*" -o -name "libcurl*" 2>/dev/null | head -3)
    echo -e "${GREEN}✓ libcurl found:${NC}"
    echo "$CURL_LIBS" | sed 's/^/  /'
else
    echo -e "${RED}✗ libcurl not found${NC}"
fi
echo ""

# Check for OpenSSL
if docker exec "$CONTAINER" find /usr/lib -name "*ssl*" -o -name "libssl*" 2>/dev/null | grep -q .; then
    SSL_LIBS=$(docker exec "$CONTAINER" find /usr/lib -name "*ssl*" -o -name "libssl*" 2>/dev/null | head -3)
    echo -e "${GREEN}✓ OpenSSL found:${NC}"
    echo "$SSL_LIBS" | sed 's/^/  /'
else
    echo -e "${RED}✗ OpenSSL not found${NC}"
fi
echo ""

# Check for zlib
if docker exec "$CONTAINER" find /usr/lib -name "*zlib*" -o -name "libz*" 2>/dev/null | grep -q .; then
    ZLIB_LIBS=$(docker exec "$CONTAINER" find /usr/lib -name "*zlib*" -o -name "libz*" 2>/dev/null | head -3)
    echo -e "${GREEN}✓ zlib found:${NC}"
    echo "$ZLIB_LIBS" | sed 's/^/  /'
else
    echo -e "${RED}✗ zlib not found${NC}"
fi
echo ""

echo -e "${BLUE}3. ML Libraries (XGBoost, LightGBM, CatBoost)${NC}"
echo "----------------------------------------"

# Check XGBoost
XGBOOST_LIBS=$(docker exec "$CONTAINER" find /usr/local/lib /usr/lib -name "*xgboost*" 2>/dev/null || true)
if [ -n "$XGBOOST_LIBS" ]; then
    echo -e "${GREEN}✓ XGBoost libraries found:${NC}"
    echo "$XGBOOST_LIBS" | sed 's/^/  /'
    # Check if library is loadable
    if docker exec "$CONTAINER" ldconfig -p 2>/dev/null | grep -q xgboost; then
        echo -e "  ${GREEN}✓ XGBoost is in ldconfig cache${NC}"
    fi
else
    echo -e "${YELLOW}⚠ XGBoost libraries not found${NC}"
fi
echo ""

# Check LightGBM
LIGHTGBM_LIBS=$(docker exec "$CONTAINER" find /usr/local/lib /usr/lib -name "*lightgbm*" 2>/dev/null || true)
if [ -n "$LIGHTGBM_LIBS" ]; then
    echo -e "${GREEN}✓ LightGBM libraries found:${NC}"
    echo "$LIGHTGBM_LIBS" | sed 's/^/  /'
    if docker exec "$CONTAINER" ldconfig -p 2>/dev/null | grep -q lightgbm; then
        echo -e "  ${GREEN}✓ LightGBM is in ldconfig cache${NC}"
    fi
else
    echo -e "${YELLOW}⚠ LightGBM libraries not found${NC}"
fi
echo ""

# Check CatBoost
CATBOOST_LIBS=$(docker exec "$CONTAINER" find /usr/local/lib /usr/lib -name "*catboost*" 2>/dev/null || true)
if [ -n "$CATBOOST_LIBS" ]; then
    echo -e "${GREEN}✓ CatBoost libraries found:${NC}"
    echo "$CATBOOST_LIBS" | sed 's/^/  /'
    if docker exec "$CONTAINER" ldconfig -p 2>/dev/null | grep -q catboost; then
        echo -e "  ${GREEN}✓ CatBoost is in ldconfig cache${NC}"
    fi
else
    echo -e "${YELLOW}⚠ CatBoost libraries not found${NC}"
fi
echo ""

echo -e "${BLUE}4. ONNX Runtime${NC}"
echo "----------------------------------------"

if docker exec "$CONTAINER" test -d /usr/local/onnxruntime 2>/dev/null; then
    echo -e "${GREEN}✓ ONNX Runtime directory exists${NC}"
    ONNX_LIBS=$(docker exec "$CONTAINER" find /usr/local/onnxruntime/lib -name "*.so*" 2>/dev/null | wc -l)
    echo "  Found $ONNX_LIBS library files"
    
    # Check for main ONNX library
    if docker exec "$CONTAINER" test -f /usr/local/onnxruntime/lib/libonnxruntime.so 2>/dev/null || \
       docker exec "$CONTAINER" find /usr/local/onnxruntime/lib -name "libonnxruntime.so*" 2>/dev/null | grep -q .; then
        echo -e "  ${GREEN}✓ libonnxruntime.so found${NC}"
    else
        echo -e "  ${YELLOW}⚠ libonnxruntime.so not found${NC}"
    fi
    
    # List some key libraries
    echo "  Key ONNX libraries:"
    docker exec "$CONTAINER" find /usr/local/onnxruntime/lib -name "*.so*" 2>/dev/null | head -5 | sed 's/^/    /'
else
    echo -e "${RED}✗ ONNX Runtime directory not found${NC}"
fi
echo ""

echo -e "${BLUE}5. OpenMP and Eigen${NC}"
echo "----------------------------------------"

# Check OpenMP
OMP_LIBS=$(docker exec "$CONTAINER" find /usr/lib -name "*omp*" -o -name "libomp*" 2>/dev/null || true)
if [ -n "$OMP_LIBS" ]; then
    echo -e "${GREEN}✓ OpenMP libraries found:${NC}"
    echo "$OMP_LIBS" | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠ OpenMP libraries not found${NC}"
fi
echo ""

# Check Eigen
EIGEN_HEADERS=$(docker exec "$CONTAINER" find /usr/include -name "eigen3" -type d 2>/dev/null || true)
if [ -n "$EIGEN_HEADERS" ]; then
    echo -e "${GREEN}✓ Eigen headers found:${NC}"
    echo "$EIGEN_HEADERS" | sed 's/^/  /'
else
    echo -e "${YELLOW}⚠ Eigen headers not found${NC}"
fi
echo ""

echo -e "${BLUE}6. NeuronDB Extension Library${NC}"
echo "----------------------------------------"

NEURONDB_SO=$(docker exec "$CONTAINER" find /usr/lib/postgresql -name "neurondb.so" 2>/dev/null || true)
if [ -n "$NEURONDB_SO" ]; then
    echo -e "${GREEN}✓ neurondb.so found:${NC}"
    echo "  $NEURONDB_SO"
    
    # Check file size and permissions
    FILE_INFO=$(docker exec "$CONTAINER" ls -lh "$NEURONDB_SO" 2>/dev/null)
    echo "  $FILE_INFO"
    
    # Check linked libraries
    echo "  Linked libraries:"
    docker exec "$CONTAINER" ldd "$NEURONDB_SO" 2>/dev/null | grep -E "(xgboost|lightgbm|catboost|onnxruntime|curl|ssl|z)" | sed 's/^/    /' || echo "    (no ML libraries linked or ldd failed)"
else
    echo -e "${RED}✗ neurondb.so not found${NC}"
fi
echo ""

echo -e "${BLUE}7. Library Path Configuration${NC}"
echo "----------------------------------------"

LD_LIBRARY_PATH=$(docker exec "$CONTAINER" printenv LD_LIBRARY_PATH 2>/dev/null || echo "not set")
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check ldconfig cache
echo "Libraries in ldconfig cache (ML-related):"
docker exec "$CONTAINER" ldconfig -p 2>/dev/null | grep -E "(xgboost|lightgbm|catboost|onnx|omp|eigen)" | sed 's/^/  /' || echo "  (none found)"
echo ""

echo -e "${BLUE}8. Functional Tests${NC}"
echo "----------------------------------------"

# Test vector operations
echo "Testing vector operations..."
VECTOR_TEST=$(docker exec "$CONTAINER" psql -U neurondb -d neurondb -tAc "SELECT '[1.0, 2.0, 3.0]'::vector;" 2>/dev/null || echo "failed")
if echo "$VECTOR_TEST" | grep -q "\[1,2,3\]"; then
    echo -e "${GREEN}✓ Vector operations working${NC}"
else
    echo -e "${YELLOW}⚠ Vector operations test: $VECTOR_TEST${NC}"
fi

# Test extension functions
echo "Testing extension version function..."
VERSION_TEST=$(docker exec "$CONTAINER" psql -U neurondb -d neurondb -tAc "SELECT neurondb.version();" 2>/dev/null || echo "failed")
if echo "$VERSION_TEST" | grep -q "version"; then
    echo -e "${GREEN}✓ Extension functions working${NC}"
    echo "  Version info: $VERSION_TEST"
else
    echo -e "${YELLOW}⚠ Extension functions test: $VERSION_TEST${NC}"
fi
echo ""

echo -e "${BLUE}9. Python Dependencies (if available)${NC}"
echo "----------------------------------------"

if docker exec "$CONTAINER" which python3 &>/dev/null; then
    PYTHON_VERSION=$(docker exec "$CONTAINER" python3 --version 2>/dev/null)
    echo "Python: $PYTHON_VERSION"
    
    # Check for CatBoost Python package
    if docker exec "$CONTAINER" python3 -c "import catboost" 2>/dev/null; then
        echo -e "${GREEN}✓ CatBoost Python package available${NC}"
    else
        echo -e "${YELLOW}⚠ CatBoost Python package not available${NC}"
    fi
else
    echo "Python3 not found in container"
fi
echo ""

echo "=========================================="
echo "Scan Complete"
echo "=========================================="





