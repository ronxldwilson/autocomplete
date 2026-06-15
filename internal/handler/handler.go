package handler

import (
	"encoding/json"
	"net/http"

	"autocomplete/internal/config"
	"autocomplete/internal/llm"
)

type Handler struct {
	client *llm.Client
	cfg    *config.Config
}

func New(client *llm.Client, cfg *config.Config) *Handler {
	return &Handler{client: client, cfg: cfg}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
