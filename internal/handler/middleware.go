package handler

import "net/http"

func APIKeyAuth(apiKey string, next http.HandlerFunc) http.HandlerFunc {
	if apiKey == "" {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-API-Key") != apiKey {
			writeError(w, http.StatusUnauthorized, "invalid or missing API key")
			return
		}
		next(w, r)
	}
}
