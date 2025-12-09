/*-------------------------------------------------------------------------
 *
 * query.go
 *    SQL query builder for NeuronMCP
 *
 * Provides utilities for building SQL queries including SELECT statements
 * and vector search queries.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/database/query.go
 *
 *-------------------------------------------------------------------------
 */

package database

import (
	"fmt"
	"strings"
)

/* QueryBuilder provides utilities for building SQL queries */
type QueryBuilder struct{}

/* Select builds a SELECT query */
func (qb *QueryBuilder) Select(table string, columns []string, where map[string]interface{}, orderBy *OrderBy, limit, offset *int) (string, []interface{}) {
	if len(columns) == 0 {
		columns = []string{"*"}
	}

	var params []interface{}
	paramIndex := 1

	selectClause := strings.Join(columns, ", ")
	fromClause := EscapeIdentifier(table)

	var whereClause string
	if len(where) > 0 {
		var conditions []string
		for key, value := range where {
			escapedKey := EscapeIdentifier(key)
			conditions = append(conditions, fmt.Sprintf("%s = $%d", escapedKey, paramIndex))
			params = append(params, value)
			paramIndex++
		}
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	var orderByClause string
	if orderBy != nil {
		orderByClause = fmt.Sprintf("ORDER BY %s %s", EscapeIdentifier(orderBy.Column), orderBy.Direction)
	}

	var limitClause string
	if limit != nil {
		limitClause = fmt.Sprintf("LIMIT $%d", paramIndex)
		params = append(params, *limit)
		paramIndex++
	}

	var offsetClause string
	if offset != nil {
		offsetClause = fmt.Sprintf("OFFSET $%d", paramIndex)
		params = append(params, *offset)
	}

	parts := []string{
		"SELECT " + selectClause,
		"FROM " + fromClause,
		whereClause,
		orderByClause,
		limitClause,
		offsetClause,
	}

	var nonEmptyParts []string
	for _, part := range parts {
		if part != "" {
			nonEmptyParts = append(nonEmptyParts, part)
		}
	}

	query := strings.Join(nonEmptyParts, " ")
	return query, params
}

/* OrderBy represents an ORDER BY clause */
type OrderBy struct {
	Column    string
	Direction string
}

/* VectorSearch builds a vector search query */
func (qb *QueryBuilder) VectorSearch(table, vectorColumn string, queryVector []float32, distanceMetric string, limit int, additionalColumns []string, minkowskiP *float64) (string, []interface{}) {
	if len(queryVector) == 0 {
		return "", nil
	}

	var params []interface{}
	paramIndex := 1

	vectorStr := formatVector(queryVector)
	params = append(params, vectorStr)
	vectorParamIndex := paramIndex
	paramIndex++

	var distanceExpr string

	switch distanceMetric {
	case "cosine":
		distanceExpr = fmt.Sprintf("%s <=> $%d::vector AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "inner_product":
		distanceExpr = fmt.Sprintf("%s <#> $%d::vector AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "l1":
		distanceExpr = fmt.Sprintf("vector_l1_distance(%s, $%d::vector) AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "hamming":
		distanceExpr = fmt.Sprintf("vector_hamming_distance(%s, $%d::vector) AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "chebyshev":
		distanceExpr = fmt.Sprintf("vector_chebyshev_distance(%s, $%d::vector) AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "minkowski":
		p := 2.0
		if minkowskiP != nil {
			p = *minkowskiP
		}
		params = append(params, p)
		pParamIndex := paramIndex
		paramIndex++
		distanceExpr = fmt.Sprintf("vector_minkowski_distance(%s, $%d::vector, $%d::double precision) AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex, pParamIndex)
	default:
		distanceExpr = fmt.Sprintf("%s <-> $%d::vector AS distance", EscapeIdentifier(vectorColumn), vectorParamIndex)
	}

	selectColumns := []string{}
	if len(additionalColumns) > 0 {
		for _, col := range additionalColumns {
			selectColumns = append(selectColumns, EscapeIdentifier(col))
		}
		selectColumns = append(selectColumns, EscapeIdentifier(vectorColumn))
	} else {
		selectColumns = append(selectColumns, "*")
	}
	selectColumns = append(selectColumns, distanceExpr)

	params = append(params, limit)
	limitParamIndex := paramIndex

	selectClause := strings.Join(selectColumns, ", ")
	query := fmt.Sprintf(
		"SELECT %s FROM %s ORDER BY distance ASC LIMIT $%d",
		selectClause,
		EscapeIdentifier(table),
		limitParamIndex,
	)

	return query, params
}

/* formatVector formats a float32 slice as a PostgreSQL vector string */
func formatVector(vec []float32) string {
	var parts []string
	for _, v := range vec {
		parts = append(parts, fmt.Sprintf("%g", v))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

