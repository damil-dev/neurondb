# Failed Workflows Summary

## Failed Workflows (10 total)

1. **Release** (Run ID: 20892784649)
   - Error: Dockerfile path not found - `lstat NeuronDB/docker: no such file or directory`
   - Issue: Workflow references `NeuronDB/docker/Dockerfile` but path structure may be incorrect

2. **Integration Tests** (Run ID: 20892776900)
   - Error: Missing PostgreSQL dev headers - `fatal error: postgres.h: No such file or directory`
   - Issue: Workflow doesn't install PostgreSQL development packages before building

3. **Chart Testing** (Run ID: 20892781399)
   - Error: YAML lint error - `too many blank lines (5 > 0)` at line 19 in `helm/neurondb/Chart.yaml`
   - Issue: Formatting issue in Chart.yaml

4. **NeuronDB - Docker** (Run ID: 20892786158)
   - Error: Compilation errors in `src/index/hnsw_am.c`
   - Issues:
     - Format security error at line 3835 (elog with ERROR)
     - Multiple warnings treated as errors (mixed declarations, return type)

5. **NeuronDB - Packages** (Run ID: 20892786608)
   - Error: Same compilation errors as NeuronDB - Docker
   - Issues: Same as above

6. **Trivy Security Scan** (Run ID: 20892783251)
   - Error: Image not found - `manifest unknown` for images with tag MAIN_WORKFLOWS
   - Issue: Images haven't been built/pushed yet (expected for test run)

7. **SBOM Generation** (Run ID: 20892782947)
   - Error: Same image not found issue as Trivy
   - Issue: Same as above

8. **Integration Tests (Full Ecosystem)** (Run ID: 20892777934)
   - Error: Same compilation errors in `hnsw_am.c`
   - Issue: Same as NeuronDB - Docker

9. **NeuronDB - Build Matrix** (Run ID: 20892779065)
   - Error: Likely same compilation errors (needs verification)

10. **Security Scan** (Run ID: 20892777412)
    - Error: Same compilation errors in `hnsw_am.c`
    - Issue: Same as NeuronDB - Docker

## Priority Fixes

1. **CRITICAL**: Fix compilation errors in `NeuronDB/src/index/hnsw_am.c` (affects multiple workflows)
2. **HIGH**: Fix Chart.yaml formatting
3. **HIGH**: Fix Release workflow Dockerfile path
4. **HIGH**: Fix Integration Tests workflow to install PostgreSQL dev packages
5. **MEDIUM**: Image scanning workflows (Trivy, SBOM) - may resolve after Docker builds succeed
