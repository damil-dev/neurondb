package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
)

/* OrganizationHandlers handles organization-related endpoints */
type OrganizationHandlers struct {
	queries *db.Queries
}

/* NewOrganizationHandlers creates new organization handlers */
func NewOrganizationHandlers(queries *db.Queries) *OrganizationHandlers {
	return &OrganizationHandlers{
		queries: queries,
	}
}

/* CreateOrganizationRequest represents a request to create an organization */
type CreateOrganizationRequest struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

/* UpdateOrganizationRequest represents a request to update an organization */
type UpdateOrganizationRequest struct {
	Name        string                 `json:"name,omitempty"`
	Description string                 `json:"description,omitempty"`
	Settings    map[string]interface{} `json:"settings,omitempty"`
}

/* CreateOrganizationMemberRequest represents a request to add a member */
type CreateOrganizationMemberRequest struct {
	UserID string `json:"user_id"`
	Role   string `json:"role"` // 'owner', 'admin', 'member', 'viewer'
}

/* CreateOrganization creates a new organization */
func (h *OrganizationHandlers) CreateOrganization(w http.ResponseWriter, r *http.Request) {
	var req CreateOrganizationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, map[string]interface{}{
			"message": "Invalid request body",
		})
		return
	}

	if req.Name == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("organization name is required"), map[string]interface{}{
			"message": "Organization name is required",
		})
		return
	}

	/* Generate slug from name */
	slug := strings.ToLower(strings.ReplaceAll(req.Name, " ", "-"))
	slug = strings.ReplaceAll(slug, "_", "-")
	/* Remove special characters */
	slug = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			return r
		}
		return -1
	}, slug)

	/* Get user ID from context (set by auth middleware) */
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Create organization */
	org := &db.Organization{
		ID:          uuid.New().String(),
		Name:        req.Name,
		Slug:        slug,
		Description: req.Description,
		Settings:    make(map[string]interface{}),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	if err := h.queries.CreateOrganization(r.Context(), org); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to create organization",
		})
		return
	}

	/* Add creator as owner */
	member := &db.OrganizationMember{
		ID:             uuid.New().String(),
		OrganizationID: org.ID,
		UserID:         userID,
		Role:           "owner",
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	if err := h.queries.CreateOrganizationMember(r.Context(), member); err != nil {
		/* Rollback organization creation */
		h.queries.DeleteOrganization(r.Context(), org.ID)
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to add member",
		})
		return
	}

	WriteSuccess(w, org, http.StatusCreated)
}

/* ListOrganizations lists organizations for the current user */
func (h *OrganizationHandlers) ListOrganizations(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	orgs, err := h.queries.ListOrganizationsForUser(r.Context(), userID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to list organizations",
		})
		return
	}

	WriteSuccess(w, orgs, http.StatusOK)
}

/* GetOrganization gets a single organization */
func (h *OrganizationHandlers) GetOrganization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is a member */
	member, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Organization not found or access denied",
		})
		return
	}

	org, err := h.queries.GetOrganization(r.Context(), orgID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Organization not found",
		})
		return
	}

	/* Include user's role in response */
	response := map[string]interface{}{
		"organization": org,
		"user_role":    member.Role,
	}

	WriteSuccess(w, response, http.StatusOK)
}

/* UpdateOrganization updates an organization */
func (h *OrganizationHandlers) UpdateOrganization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is admin or owner */
	member, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil || (member.Role != "admin" && member.Role != "owner") {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), map[string]interface{}{
			"message": "Access denied",
		})
		return
	}

	var req UpdateOrganizationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, map[string]interface{}{
			"message": "Invalid request body",
		})
		return
	}

	org, err := h.queries.GetOrganization(r.Context(), orgID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Organization not found",
		})
		return
	}

	/* Update fields */
	if req.Name != "" {
		org.Name = req.Name
	}
	if req.Description != "" {
		org.Description = req.Description
	}
	if req.Settings != nil {
		org.Settings = req.Settings
	}
	org.UpdatedAt = time.Now()

	if err := h.queries.UpdateOrganization(r.Context(), org); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to update organization",
		})
		return
	}

	WriteSuccess(w, org, http.StatusOK)
}

/* DeleteOrganization deletes an organization */
func (h *OrganizationHandlers) DeleteOrganization(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is owner */
	member, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil || member.Role != "owner" {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("only organization owners can delete organizations"), map[string]interface{}{
			"message": "Only organization owners can delete organizations",
		})
		return
	}

	if err := h.queries.DeleteOrganization(r.Context(), orgID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to delete organization",
		})
		return
	}

	WriteSuccess(w, map[string]string{"message": "Organization deleted"}, http.StatusOK)
}

/* ListMembers lists members of an organization */
func (h *OrganizationHandlers) ListMembers(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is a member */
	_, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Organization not found or access denied",
		})
		return
	}

	members, err := h.queries.ListOrganizationMembers(r.Context(), orgID)
	if err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to list members",
		})
		return
	}

	WriteSuccess(w, members, http.StatusOK)
}

/* AddMember adds a member to an organization */
func (h *OrganizationHandlers) AddMember(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is admin or owner */
	member, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil || (member.Role != "admin" && member.Role != "owner") {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), map[string]interface{}{
			"message": "Access denied",
		})
		return
	}

	var req CreateOrganizationMemberRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, map[string]interface{}{
			"message": "Invalid request body",
		})
		return
	}

	/* Validate role */
	validRoles := map[string]bool{"owner": true, "admin": true, "member": true, "viewer": true}
	if !validRoles[req.Role] {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid role"), map[string]interface{}{
			"message": "Invalid role. Must be: owner, admin, member, or viewer",
		})
		return
	}

	/* Only owners can add owners */
	if req.Role == "owner" && member.Role != "owner" {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("only owners can add other owners"), map[string]interface{}{
			"message": "Only owners can add other owners",
		})
		return
	}

	newMember := &db.OrganizationMember{
		ID:             uuid.New().String(),
		OrganizationID: orgID,
		UserID:         req.UserID,
		Role:           req.Role,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
	}

	if err := h.queries.CreateOrganizationMember(r.Context(), newMember); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to add member",
		})
		return
	}

	WriteSuccess(w, newMember, http.StatusCreated)
}

/* UpdateMember updates a member's role */
func (h *OrganizationHandlers) UpdateMember(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]
	memberID := vars["member_id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is admin or owner */
	member, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil || (member.Role != "admin" && member.Role != "owner") {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), map[string]interface{}{
			"message": "Access denied",
		})
		return
	}

	var req struct {
		Role string `json:"role"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, err, map[string]interface{}{
			"message": "Invalid request body",
		})
		return
	}

	/* Validate role */
	validRoles := map[string]bool{"owner": true, "admin": true, "member": true, "viewer": true}
	if !validRoles[req.Role] {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid role"), map[string]interface{}{
			"message": "Invalid role. Must be: owner, admin, member, or viewer",
		})
		return
	}

	/* Get member to update */
	targetMember, err := h.queries.GetOrganizationMemberByID(r.Context(), memberID)
	if err != nil || targetMember.OrganizationID != orgID {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Member not found",
		})
		return
	}

	/* Only owners can change roles to owner */
	if req.Role == "owner" && member.Role != "owner" {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("only owners can assign owner role"), map[string]interface{}{
			"message": "Only owners can assign owner role",
		})
		return
	}

	targetMember.Role = req.Role
	targetMember.UpdatedAt = time.Now()

	if err := h.queries.UpdateOrganizationMember(r.Context(), targetMember); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to update member",
		})
		return
	}

	WriteSuccess(w, targetMember, http.StatusOK)
}

/* RemoveMember removes a member from an organization */
func (h *OrganizationHandlers) RemoveMember(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	orgID := vars["id"]
	memberID := vars["member_id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok || userID == "" {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("user not authenticated"), map[string]interface{}{
			"message": "User not authenticated",
		})
		return
	}

	/* Check if user is admin or owner */
	member, err := h.queries.GetOrganizationMember(r.Context(), orgID, userID)
	if err != nil || (member.Role != "admin" && member.Role != "owner") {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), map[string]interface{}{
			"message": "Access denied",
		})
		return
	}

	/* Get member to remove */
	targetMember, err := h.queries.GetOrganizationMemberByID(r.Context(), memberID)
	if err != nil || targetMember.OrganizationID != orgID {
		WriteError(w, r, http.StatusNotFound, err, map[string]interface{}{
			"message": "Member not found",
		})
		return
	}

	/* Prevent removing the last owner */
	if targetMember.Role == "owner" {
		owners, _ := h.queries.ListOrganizationMembersByRole(r.Context(), orgID, "owner")
		if len(owners) <= 1 {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("cannot remove the last owner"), map[string]interface{}{
				"message": "Cannot remove the last owner",
			})
			return
		}
	}

	if err := h.queries.DeleteOrganizationMember(r.Context(), memberID); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, map[string]interface{}{
			"message": "Failed to remove member",
		})
		return
	}

	WriteSuccess(w, map[string]string{"message": "Member removed"}, http.StatusOK)
}

