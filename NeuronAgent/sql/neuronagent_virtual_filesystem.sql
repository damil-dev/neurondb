-- Virtual File System for Agent Scratchpad
-- Provides persistent file storage for agent operations and data externalization
-- Supports both database storage for small files and object storage for large files

-- Virtual files table
CREATE TABLE IF NOT EXISTS neurondb_agent.virtual_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    content BYTEA,
    content_s3_key TEXT,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    size BIGINT NOT NULL DEFAULT 0,
    compressed BOOLEAN NOT NULL DEFAULT false,
    storage_backend TEXT NOT NULL DEFAULT 'database' CHECK (storage_backend IN ('database', 's3')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, path)
);

-- Virtual directories table
CREATE TABLE IF NOT EXISTS neurondb_agent.virtual_directories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, path)
);

-- File access log for audit and analytics
CREATE TABLE IF NOT EXISTS neurondb_agent.file_access_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES neurondb_agent.virtual_files(id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    operation TEXT NOT NULL CHECK (operation IN ('read', 'write', 'delete', 'create', 'copy', 'move')),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for virtual_files
CREATE INDEX IF NOT EXISTS idx_virtual_files_agent_id ON neurondb_agent.virtual_files(agent_id);
CREATE INDEX IF NOT EXISTS idx_virtual_files_session_id ON neurondb_agent.virtual_files(session_id);
CREATE INDEX IF NOT EXISTS idx_virtual_files_path ON neurondb_agent.virtual_files(agent_id, path);
CREATE INDEX IF NOT EXISTS idx_virtual_files_storage_backend ON neurondb_agent.virtual_files(storage_backend);
CREATE INDEX IF NOT EXISTS idx_virtual_files_created_at ON neurondb_agent.virtual_files(created_at DESC);

-- Indexes for virtual_directories
CREATE INDEX IF NOT EXISTS idx_virtual_directories_agent_id ON neurondb_agent.virtual_directories(agent_id);
CREATE INDEX IF NOT EXISTS idx_virtual_directories_session_id ON neurondb_agent.virtual_directories(session_id);
CREATE INDEX IF NOT EXISTS idx_virtual_directories_path ON neurondb_agent.virtual_directories(agent_id, path);

-- Indexes for file_access_log
CREATE INDEX IF NOT EXISTS idx_file_access_log_file_id ON neurondb_agent.file_access_log(file_id);
CREATE INDEX IF NOT EXISTS idx_file_access_log_agent_id ON neurondb_agent.file_access_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_file_access_log_accessed_at ON neurondb_agent.file_access_log(accessed_at DESC);

COMMENT ON TABLE neurondb_agent.virtual_files IS 'Virtual file system for agent scratchpad operations. Stores file content in database for small files or S3 for large files.';
COMMENT ON TABLE neurondb_agent.virtual_directories IS 'Virtual directory structure for organizing agent files.';
COMMENT ON TABLE neurondb_agent.file_access_log IS 'Audit log of all file operations for security and analytics.';

COMMENT ON COLUMN neurondb_agent.virtual_files.content IS 'File content stored in database for files < 1MB. NULL if stored in S3.';
COMMENT ON COLUMN neurondb_agent.virtual_files.content_s3_key IS 'S3 object key for files stored in object storage. NULL if stored in database.';
COMMENT ON COLUMN neurondb_agent.virtual_files.storage_backend IS 'Storage backend used: database for small files, s3 for large files.';
COMMENT ON COLUMN neurondb_agent.virtual_files.compressed IS 'Whether file content is compressed using gzip.';
