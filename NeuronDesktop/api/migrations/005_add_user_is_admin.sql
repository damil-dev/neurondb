-- Add is_admin flag to users for role-based access control
ALTER TABLE users
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

-- Index for faster role checks (optional)
CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);








