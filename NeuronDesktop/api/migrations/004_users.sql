-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Update profiles to reference users (user_id is TEXT, so we can't add a foreign key constraint directly)
-- Profiles will continue to use user_id as TEXT for now, matching existing user accounts

-- Note: Default admin user should be created programmatically with proper password hashing
-- A default admin user can be created via the registration endpoint or a setup script

