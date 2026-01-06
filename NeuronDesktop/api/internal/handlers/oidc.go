package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/auth/oidc"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/session"
	"golang.org/x/crypto/bcrypt"
)

/* OIDCHandlers handles OIDC authentication */
type OIDCHandlers struct {
	provider      *oidc.Provider
	sessionMgr    *session.Manager
	queries       *db.Queries
	issuer        string
	loginAttempts map[string]*oidc.LoginAttempt
}

/* NewOIDCHandlers creates new OIDC handlers */
func NewOIDCHandlers(provider *oidc.Provider, sessionMgr *session.Manager, queries *db.Queries, issuer string) *OIDCHandlers {
	return &OIDCHandlers{
		provider:      provider,
		sessionMgr:    sessionMgr,
		queries:       queries,
		issuer:        issuer,
		loginAttempts: make(map[string]*oidc.LoginAttempt),
	}
}

/* StartOIDCFlow initiates OIDC login flow */
func (h *OIDCHandlers) StartOIDCFlow(w http.ResponseWriter, r *http.Request) {
	redirectURI := r.URL.Query().Get("redirect_uri")
	if redirectURI == "" {
		redirectURI = "/"
	}

	attempt, err := oidc.NewLoginAttempt(10 * time.Minute)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create login attempt: %w", err), nil)
		return
	}

	attempt.RedirectURI = redirectURI

	h.loginAttempts[attempt.State] = attempt

	authURL, err := h.provider.AuthCodeURL(attempt.State, attempt.Nonce, attempt.CodeVerifier)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to generate auth URL: %w", err), nil)
		return
	}

	acceptHeader := r.Header.Get("Accept")
	if strings.Contains(acceptHeader, "text/html") {
		http.Redirect(w, r, authURL, http.StatusFound)
		return
	}

	response := map[string]interface{}{
		"auth_url":     authURL,
		"state":        attempt.State,
		"redirect_uri": redirectURI,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

/* OIDCCallback handles OIDC callback */
func (h *OIDCHandlers) OIDCCallback(w http.ResponseWriter, r *http.Request) {
	code := r.URL.Query().Get("code")
	state := r.URL.Query().Get("state")

	if code == "" || state == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("missing code or state"), nil)
		return
	}

	attempt, ok := h.loginAttempts[state]
	if !ok {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid or expired state"), nil)
		return
	}

	delete(h.loginAttempts, state)

	if time.Now().After(attempt.ExpiresAt) {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("login attempt expired"), nil)
		return
	}

	oauth2Token, err := h.provider.ExchangeCode(r.Context(), code, attempt.CodeVerifier)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to exchange code: %w", err), nil)
		return
	}

	rawIDToken, ok := oauth2Token.Extra("id_token").(string)
	if !ok {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("missing id_token"), nil)
		return
	}

	idToken, rawClaims, err := h.provider.VerifyIDToken(r.Context(), rawIDToken)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to verify ID token: %w", err), nil)
		return
	}

	if idToken.Nonce != attempt.Nonce {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("nonce mismatch"), nil)
		return
	}

	claims := oidc.ExtractClaims(rawClaims)

	user, err := h.getOrCreateUser(r.Context(), claims)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to get or create user: %w", err), nil)
		return
	}

	userAgent := r.Header.Get("User-Agent")
	ip := getClientIP(r)
	sess, refreshToken, err := h.sessionMgr.CreateSession(r.Context(), user.ID, userAgent, ip)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to create session: %w", err), nil)
		return
	}

	h.sessionMgr.SetCookies(w, sess.ID, refreshToken)

	var profileID string
	if profile, err := h.queries.GetDefaultProfileForUser(r.Context(), user.ID); err == nil && profile != nil {
		profileID = profile.ID
	}

	redirectURL := attempt.RedirectURI
	if redirectURL == "" {
		redirectURL = "/"
	}

	if profileID != "" {
		if strings.Contains(redirectURL, "?") {
			redirectURL += "&profile_id=" + profileID
		} else {
			redirectURL += "?profile_id=" + profileID
		}
	}

	acceptHeader := r.Header.Get("Accept")
	if strings.Contains(acceptHeader, "text/html") || redirectURL != "/" {
		http.Redirect(w, r, redirectURL, http.StatusFound)
		return
	}

	response := map[string]interface{}{
		"user_id":    user.ID,
		"username":   user.Username,
		"is_admin":   user.IsAdmin,
		"session_id": sess.ID,
	}
	if profileID != "" {
		response["profile_id"] = profileID
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

/* Logout handles logout */
func (h *OIDCHandlers) Logout(w http.ResponseWriter, r *http.Request) {
	sess, ok := session.GetSessionFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("not authenticated"), nil)
		return
	}

	if err := h.sessionMgr.RevokeSession(r.Context(), sess.ID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to revoke session: %w", err), nil)
		return
	}

	h.sessionMgr.ClearCookies(w)

	WriteSuccess(w, map[string]interface{}{"message": "logged out"}, http.StatusOK)
}

/* RefreshToken handles refresh token endpoint */
func (h *OIDCHandlers) RefreshToken(w http.ResponseWriter, r *http.Request) {
	refreshToken := h.sessionMgr.GetRefreshTokenFromRequest(r)
	if refreshToken == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("missing refresh token"), nil)
		return
	}

	sess, accessToken, newRefreshToken, err := h.sessionMgr.RefreshSession(r.Context(), refreshToken)
	if err != nil {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("failed to refresh session: %w", err), nil)
		return
	}

	h.sessionMgr.SetCookies(w, accessToken, newRefreshToken)

	response := map[string]interface{}{
		"session_id": sess.ID,
		"user_id":    sess.UserID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *OIDCHandlers) getOrCreateUser(ctx context.Context, claims *oidc.Claims) (*db.User, error) {
	var oidcIdentity struct {
		UserID string
	}

	query := `
		SELECT user_id FROM oidc_identities
		WHERE issuer = $1 AND subject = $2
	`
	err := h.queries.GetDB().QueryRowContext(ctx, query, h.issuer, claims.Subject).Scan(&oidcIdentity.UserID)
	if err == nil {
		user, err := h.queries.GetUserByID(ctx, oidcIdentity.UserID)
		if err == nil {
			updateQuery := `
				UPDATE oidc_identities
				SET email = $1, name = $2, updated_at = NOW()
				WHERE issuer = $3 AND subject = $4
			`
			h.queries.GetDB().ExecContext(ctx, updateQuery, claims.Email, claims.Name, h.issuer, claims.Subject)
			return user, nil
		}
	}

	username := claims.PreferredUsername
	if username == "" {
		username = claims.Email
	}
	if username == "" {
		username = claims.Subject
	}

	passwordHash, err := bcrypt.GenerateFromPassword([]byte(fmt.Sprintf("oidc-%s-%d", claims.Subject, time.Now().Unix())), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}

	user := &db.User{
		Username:     username,
		PasswordHash: string(passwordHash),
		IsAdmin:      false,
	}

	if err := h.queries.CreateUser(ctx, user); err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	insertQuery := `
		INSERT INTO oidc_identities (issuer, subject, user_id, email, name)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (issuer, subject) DO UPDATE
		SET email = EXCLUDED.email, name = EXCLUDED.name, updated_at = NOW()
	`
	_, err = h.queries.GetDB().ExecContext(ctx, insertQuery, h.issuer, claims.Subject, user.ID, claims.Email, claims.Name)
	if err != nil {
		return nil, fmt.Errorf("failed to create OIDC identity: %w", err)
	}

	return user, nil
}

func getClientIP(r *http.Request) string {
	forwarded := r.Header.Get("X-Forwarded-For")
	if forwarded != "" {
		return forwarded
	}

	realIP := r.Header.Get("X-Real-IP")
	if realIP != "" {
		return realIP
	}

	return r.RemoteAddr
}
