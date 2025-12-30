-- Migration: Add profile username and password fields
-- This allows profiles to have their own login credentials
-- When users log in with profile credentials, they are automatically on that profile

-- Add profile_username column (nullable, unique)
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS profile_username TEXT;

-- Add profile_password_hash column (nullable, stores bcrypt hash)
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS profile_password_hash TEXT;

-- Create index on profile_username for fast lookups
CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(profile_username) WHERE profile_username IS NOT NULL;

-- Add comment
COMMENT ON COLUMN profiles.profile_username IS 'Username for this profile. When user logs in with this username/password, they are automatically on this profile.';
COMMENT ON COLUMN profiles.profile_password_hash IS 'Bcrypt hash of the profile password. Never store plain text passwords.';





