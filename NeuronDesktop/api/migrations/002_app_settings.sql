-- App settings table for storing application-level configuration
-- Used for wizard completion state, feature flags, etc.

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_settings_updated_at ON app_settings(updated_at);

-- Insert default settings if needed
-- (No default values for wizard completion - starts as not completed)







