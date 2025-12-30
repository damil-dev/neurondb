package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds application configuration
type Config struct {
	Database DatabaseConfig
	Server   ServerConfig
	Logging  LoggingConfig
	CORS     CORSConfig
	Auth     AuthConfig
	Session  SessionConfig
	Security SecurityConfig
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Host            string
	Port            string
	User            string
	Password        string
	Name            string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Host         string
	Port         string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

// LoggingConfig holds logging configuration
type LoggingConfig struct {
	Level  string
	Format string
	Output string
}

// CORSConfig holds CORS configuration
type CORSConfig struct {
	AllowedOrigins []string
	AllowedMethods []string
	AllowedHeaders []string
}

// AuthConfig holds authentication configuration
type AuthConfig struct {
	Mode            string // "oidc", "jwt", "hybrid"
	OIDC            OIDCConfig
	JWTSecret       string
	EnableLocalAuth bool
}

// OIDCConfig holds OIDC configuration
type OIDCConfig struct {
	IssuerURL    string
	ClientID     string
	ClientSecret string
	RedirectURL  string
	Scopes       []string
}

// SessionConfig holds session configuration
type SessionConfig struct {
	CookieDomain   string
	CookieSecure   bool
	CookieSameSite string // "Lax", "Strict", "None"
	AccessTTL      time.Duration
	RefreshTTL     time.Duration
}

// SecurityConfig holds security-related configuration
type SecurityConfig struct {
	EnableSQLConsole bool // Enable arbitrary SQL execution endpoint (default: false)
}

// Load loads configuration from environment variables
func Load() *Config {
	return &Config{
		Database: DatabaseConfig{
			Host:            getEnv("DB_HOST", "localhost"),
			Port:            getEnv("DB_PORT", "5432"),
			User:            getEnv("DB_USER", "neurondesk"),
			Password:        getEnv("DB_PASSWORD", "neurondesk"),
			Name:            getEnv("DB_NAME", "neurondesk"),
			MaxOpenConns:    getEnvInt("DB_MAX_OPEN_CONNS", 25),
			MaxIdleConns:    getEnvInt("DB_MAX_IDLE_CONNS", 5),
			ConnMaxLifetime: getEnvDuration("DB_CONN_MAX_LIFETIME", 5*time.Minute),
		},
		Server: ServerConfig{
			Host:         getEnv("SERVER_HOST", "0.0.0.0"),
			Port:         getEnv("SERVER_PORT", "8081"),
			ReadTimeout:  getEnvDuration("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout: getEnvDuration("SERVER_WRITE_TIMEOUT", 30*time.Second),
		},
		Logging: LoggingConfig{
			Level:  getEnv("LOG_LEVEL", "info"),
			Format: getEnv("LOG_FORMAT", "json"),
			Output: getEnv("LOG_OUTPUT", "stdout"),
		},
		CORS: CORSConfig{
			AllowedOrigins: getEnvSlice("CORS_ALLOWED_ORIGINS", []string{"*"}),
			AllowedMethods: getEnvSlice("CORS_ALLOWED_METHODS", []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
			AllowedHeaders: getEnvSlice("CORS_ALLOWED_HEADERS", []string{"Content-Type", "Authorization"}),
		},
		Auth: AuthConfig{
			Mode:            getEnv("AUTH_MODE", "oidc"),
			JWTSecret:       getEnv("JWT_SECRET", ""),
			EnableLocalAuth: getEnv("ENABLE_LOCAL_AUTH", "true") == "true",
			OIDC: OIDCConfig{
				IssuerURL:    getEnv("OIDC_ISSUER_URL", ""),
				ClientID:     getEnv("OIDC_CLIENT_ID", ""),
				ClientSecret: getEnv("OIDC_CLIENT_SECRET", ""),
				RedirectURL:  getEnv("OIDC_REDIRECT_URL", ""),
				Scopes:       getEnvSlice("OIDC_SCOPES", []string{"openid", "profile", "email"}),
			},
		},
		Session: SessionConfig{
			CookieDomain:   getEnv("SESSION_COOKIE_DOMAIN", ""),
			CookieSecure:   getEnv("SESSION_SECURE", "true") == "true",
			CookieSameSite: getEnv("SESSION_SAMESITE", "Lax"),
			AccessTTL:      getEnvDuration("SESSION_ACCESS_TTL", 15*time.Minute),
			RefreshTTL:     getEnvDuration("SESSION_REFRESH_TTL", 30*24*time.Hour),
		},
		Security: SecurityConfig{
			EnableSQLConsole: getEnv("ENABLE_SQL_CONSOLE", "false") == "true",
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getEnvSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		// Simple comma-separated parsing
		parts := []string{}
		for _, part := range splitString(value, ",") {
			parts = append(parts, trimSpace(part))
		}
		if len(parts) > 0 {
			return parts
		}
	}
	return defaultValue
}

func splitString(s, sep string) []string {
	parts := []string{}
	current := ""
	for _, char := range s {
		if string(char) == sep {
			if current != "" {
				parts = append(parts, current)
				current = ""
			}
		} else {
			current += string(char)
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	return parts
}

func trimSpace(s string) string {
	start := 0
	end := len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t') {
		end--
	}
	return s[start:end]
}

// DSN returns the database connection string
func (c *DatabaseConfig) DSN() string {
	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		c.Host, c.Port, c.User, c.Password, c.Name)
}
