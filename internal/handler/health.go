package handler

import "net/http"

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	if err := h.client.Health(); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"status": "unhealthy",
			"error":  err.Error(),
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}
