package oidc

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"github.com/google/uuid"
	"golang.org/x/oauth2"
)

/* Provider wraps OIDC provider and OAuth2 config */
type Provider struct {
	provider   *oidc.Provider
	oauth2Conf *oauth2.Config
	verifier   *oidc.IDTokenVerifier
}

/* NewProvider creates a new OIDC provider */
func NewProvider(ctx context.Context, issuerURL, clientID, clientSecret, redirectURL string, scopes []string) (*Provider, error) {
	provider, err := oidc.NewProvider(ctx, issuerURL)
	if err != nil {
		return nil, fmt.Errorf("failed to create OIDC provider: %w", err)
	}

	oauth2Conf := &oauth2.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURL:  redirectURL,
		Scopes:       scopes,
		Endpoint:     provider.Endpoint(),
	}

	verifier := provider.Verifier(&oidc.Config{
		ClientID: clientID,
	})

	return &Provider{
		provider:   provider,
		oauth2Conf: oauth2Conf,
		verifier:   verifier,
	}, nil
}

/* AuthCodeURL generates the OAuth2 authorization URL with PKCE */
func (p *Provider) AuthCodeURL(state, nonce, codeVerifier string) (string, error) {
	codeChallenge := base64URLEncode(sha256Hash(codeVerifier))

	opts := []oauth2.AuthCodeOption{
		oauth2.SetAuthURLParam("code_challenge", codeChallenge),
		oauth2.SetAuthURLParam("code_challenge_method", "S256"),
		oauth2.SetAuthURLParam("nonce", nonce),
	}

	url := p.oauth2Conf.AuthCodeURL(state, opts...)
	return url, nil
}

/* ExchangeCode exchanges authorization code for tokens */
func (p *Provider) ExchangeCode(ctx context.Context, code, codeVerifier string) (*oauth2.Token, error) {
	opts := []oauth2.AuthCodeOption{
		oauth2.SetAuthURLParam("code_verifier", codeVerifier),
	}

	token, err := p.oauth2Conf.Exchange(ctx, code, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}

	return token, nil
}

/* VerifyIDToken verifies and extracts claims from ID token */
func (p *Provider) VerifyIDToken(ctx context.Context, rawIDToken string) (*oidc.IDToken, map[string]interface{}, error) {
	idToken, err := p.verifier.Verify(ctx, rawIDToken)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to verify ID token: %w", err)
	}

	var claims map[string]interface{}
	if err := idToken.Claims(&claims); err != nil {
		return nil, nil, fmt.Errorf("failed to extract claims: %w", err)
	}

	return idToken, claims, nil
}

/* Claims represents OIDC user claims */
type Claims struct {
	Subject           string `json:"sub"`
	Email             string `json:"email"`
	EmailVerified     bool   `json:"email_verified"`
	Name              string `json:"name"`
	PreferredUsername string `json:"preferred_username"`
	GivenName         string `json:"given_name"`
	FamilyName        string `json:"family_name"`
	Picture           string `json:"picture"`
}

/* ExtractClaims extracts structured claims from raw claims map */
func ExtractClaims(rawClaims map[string]interface{}) *Claims {
	claims := &Claims{}

	if sub, ok := rawClaims["sub"].(string); ok {
		claims.Subject = sub
	}
	if email, ok := rawClaims["email"].(string); ok {
		claims.Email = email
	}
	if emailVerified, ok := rawClaims["email_verified"].(bool); ok {
		claims.EmailVerified = emailVerified
	}
	if name, ok := rawClaims["name"].(string); ok {
		claims.Name = name
	}
	if preferredUsername, ok := rawClaims["preferred_username"].(string); ok {
		claims.PreferredUsername = preferredUsername
	}
	if givenName, ok := rawClaims["given_name"].(string); ok {
		claims.GivenName = givenName
	}
	if familyName, ok := rawClaims["family_name"].(string); ok {
		claims.FamilyName = familyName
	}
	if picture, ok := rawClaims["picture"].(string); ok {
		claims.Picture = picture
	}

	return claims
}

/* LoginAttempt represents a stored login attempt with state/nonce */
type LoginAttempt struct {
	ID           string
	State        string
	Nonce        string
	CodeVerifier string
	RedirectURI  string
	CreatedAt    time.Time
	ExpiresAt    time.Time
}

/* NewLoginAttempt creates a new login attempt */
func NewLoginAttempt(ttl time.Duration) (*LoginAttempt, error) {
	state, err := generateRandomString(32)
	if err != nil {
		return nil, err
	}

	nonce, err := generateRandomString(32)
	if err != nil {
		return nil, err
	}

	codeVerifier, err := generateCodeVerifier()
	if err != nil {
		return nil, err
	}

	now := time.Now()
	return &LoginAttempt{
		ID:           uuid.New().String(),
		State:        state,
		Nonce:        nonce,
		CodeVerifier: codeVerifier,
		CreatedAt:    now,
		ExpiresAt:    now.Add(ttl),
	}, nil
}

/* Helper functions */

func generateRandomString(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(bytes)[:length], nil
}

func generateCodeVerifier() (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(bytes), nil
}

func base64URLEncode(data []byte) string {
	return base64.RawURLEncoding.EncodeToString(data)
}

func sha256Hash(data string) []byte {
	h := sha256.Sum256([]byte(data))
	return h[:]
}
