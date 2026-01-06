#!/bin/bash

# Update all version numbers to 2.0.0 across the entire codebase

cd /Users/pgedge/pge/neurondb2 || exit 1

echo "Updating all versions to 2.0.0..."
echo ""

# 1. NeuronDB control files
echo "1. Updating NeuronDB control files..."
find . -name "neurondb.control" -type f | while read -r file; do
    echo "  - $file"
    sed -i '' "s/default_version = '1\.0'/default_version = '2.0'/g" "$file"
done

# 2. Package.json files (NeuronMCP, NeuronDesktop)
echo "2. Updating package.json files..."
find . -name "package.json" -type f | grep -v node_modules | grep -v ".next" | while read -r file; do
    echo "  - $file"
    sed -i '' 's/"version": "1\.0\.0"/"version": "2.0.0"/g' "$file"
    sed -i '' 's/"version": "2\.0\.0"/"version": "2.0.0"/g' "$file"  # In case already 2.0.0
done

# 3. Helm charts
echo "3. Updating Helm charts..."
find . -name "Chart.yaml" -type f | while read -r file; do
    echo "  - $file"
    sed -i '' 's/^version: 1\.0\.0/version: 2.0.0/g' "$file"
    sed -i '' 's/^appVersion: "1\.0\.0"/appVersion: "2.0.0"/g' "$file"
done

# 4. OpenAPI specs
echo "4. Updating OpenAPI specs..."
find . -name "openapi.yaml" -o -name "openapi.yml" | while read -r file; do
    echo "  - $file"
    sed -i '' 's/version: 1\.0\.0/version: 2.0.0/g' "$file"
done

# 5. Go files with version strings
echo "5. Updating Go version strings..."
find . -name "*.go" -type f | grep -v vendor | while read -r file; do
    if grep -q '"1\.0\.0"' "$file" 2>/dev/null; then
        echo "  - $file"
        sed -i '' 's/"1\.0\.0"/"2.0.0"/g' "$file"
    fi
    if grep -q 'version:.*"1\.0\.0"' "$file" 2>/dev/null; then
        sed -i '' 's/version:.*"1\.0\.0"/version:     "2.0.0"/g' "$file"
    fi
done

# 6. Dockerfiles
echo "6. Updating Dockerfiles..."
find . -name "Dockerfile*" -type f | while read -r file; do
    if grep -q "VERSION.*1\.0" "$file" 2>/dev/null; then
        echo "  - $file"
        sed -i '' 's/VERSION=1\.0/VERSION=2.0/g' "$file"
        sed -i '' 's/VERSION 1\.0/VERSION 2.0/g' "$file"
    fi
done

# 7. Docker compose files
echo "7. Updating docker-compose files..."
find . -name "docker-compose*.yml" -type f | while read -r file; do
    if grep -q ":1\.0" "$file" 2>/dev/null; then
        echo "  - $file"
        sed -i '' 's/:1\.0/:2.0/g' "$file"
    fi
done

# 8. SQL files with version references
echo "8. Updating SQL version files..."
for sqlfile in NeuronDB/neurondb--1.0.sql*; do
    if [ -f "$sqlfile" ]; then
        newfile="${sqlfile/1.0/2.0}"
        echo "  - Renaming $sqlfile to $newfile"
        git mv "$sqlfile" "$newfile" 2>/dev/null || mv "$sqlfile" "$newfile"
    fi
done

# 9. Create upgrade script
echo "9. Creating upgrade SQL script..."
cat > NeuronDB/neurondb--1.0--2.0.sql << 'EOF'
-- Upgrade script from NeuronDB 1.0 to 2.0
-- This file is used by PostgreSQL ALTER EXTENSION ... UPDATE TO

-- Add any schema changes here
-- For now, this is a compatibility upgrade with no schema changes

EOF

echo ""
echo "Version update complete!"
echo "All modules updated to 2.0.0"

