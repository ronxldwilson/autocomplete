package main

import (
	"context"
	"embed"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"autocomplete/internal/config"
	"autocomplete/internal/handler"
	"autocomplete/internal/llm"
)

//go:embed static
var staticFiles embed.FS

func main() {
	cfg := config.Load()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	srv := llm.NewServer(llm.ServerConfig{
		BinPath:   cfg.LlamaServerBin,
		ModelPath: cfg.ModelPath,
		Port:      cfg.LlamaPort,
		Threads:   cfg.Threads,
		CtxSize:   cfg.CtxSize,
	})

	if err := srv.Start(ctx); err != nil {
		log.Fatalf("Failed to start llama-server: %v", err)
	}
	defer srv.Stop()

	client := llm.NewClient(srv.BaseURL())
	h := handler.New(client, cfg)

	staticFS, _ := fs.Sub(staticFiles, "static")

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("POST /autocomplete", handler.APIKeyAuth(cfg.APIKey, h.Autocomplete))
	mux.Handle("GET /", http.FileServer(http.FS(staticFS)))

	httpServer := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		log.Println("shutting down...")
		cancel()
		httpServer.Shutdown(context.Background())
	}()

	log.Printf("autocomplete service listening on :%s", cfg.Port)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}
