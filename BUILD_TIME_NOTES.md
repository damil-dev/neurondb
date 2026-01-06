# Docker Build Time Notes

## Expected Build Times

**First-time builds can take 10-30+ minutes** because NeuronDB compiles from source:

1. **NeuronDB (CPU)**: 15-30 minutes
   - Downloads Debian packages (build-essential, PostgreSQL dev headers, etc.)
   - Compiles C/C++ PostgreSQL extension from source
   - Creates optimized binaries

2. **NeuronAgent**: 3-5 minutes
   - Compiles Go binary
   - Faster than NeuronDB

3. **NeuronMCP**: 3-5 minutes  
   - Compiles Go binary
   - Faster than NeuronDB

4. **NeuronDesktop**: 5-10 minutes
   - Compiles Go backend
   - Builds Node.js frontend
   - Longer than other Go services

## Why It Takes Time

NeuronDB is a **PostgreSQL extension written in C/C++** that:
- Compiles native code for your platform
- Links against PostgreSQL's extension API
- Builds GPU backends (CUDA/ROCm/Metal) when applicable
- Includes ML libraries and optimizations

This is **normal and expected** - the compilation ensures optimal performance.

## Tips to Speed Up

1. **First build**: Be patient (15-30 min for NeuronDB)
2. **Subsequent builds**: Much faster (uses Docker layer cache)
3. **Rebuilds**: Only changed layers rebuild
4. **Parallel builds**: `docker compose --profile cpu build` builds all services in parallel

## Check Build Progress

```bash
# Watch build output
docker compose --profile cpu build neurondb

# Or in background (logs available via docker logs)
docker compose --profile cpu up -d --build neurondb

# Check if image exists
docker images | grep neurondb

# Check build processes
docker buildx ls
```

## Once Built

After the first build completes, starting services is fast (< 1 minute):

```bash
# Start without rebuilding (uses cached image)
docker compose --profile cpu up -d neurondb

# Verify it's running
docker compose --profile cpu ps neurondb

# Check logs
docker compose --profile cpu logs neurondb
```

