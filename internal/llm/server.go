package llm

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"time"
)

type ServerConfig struct {
	BinPath   string
	ModelPath string
	Port      string
	Threads   int
	CtxSize   int
}

type Server struct {
	cfg     ServerConfig
	cmd     *exec.Cmd
	ctx     context.Context
	baseURL string
}

func NewServer(cfg ServerConfig) *Server {
	return &Server{
		cfg:     cfg,
		baseURL: "http://127.0.0.1:" + cfg.Port,
	}
}

func (s *Server) Start(ctx context.Context) error {
	s.ctx = ctx
	return s.startProcess()
}

func (s *Server) startProcess() error {
	s.cmd = exec.CommandContext(s.ctx, s.cfg.BinPath,
		"-m", s.cfg.ModelPath,
		"-t", strconv.Itoa(s.cfg.Threads),
		"--port", s.cfg.Port,
		"--host", "127.0.0.1",
		"-c", strconv.Itoa(s.cfg.CtxSize),
		"--parallel", "3",
		"--log-disable",
	)
	s.cmd.Stdout = os.Stdout
	s.cmd.Stderr = os.Stderr

	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}

	log.Printf("llama-server started (pid %d) on port %s", s.cmd.Process.Pid, s.cfg.Port)

	// Monitor and auto-restart on crash
	go func() {
		for {
			err := s.cmd.Wait()
			if s.ctx.Err() != nil {
				return // shutting down
			}
			log.Printf("llama-server exited unexpectedly: %v — restarting in 2s", err)
			time.Sleep(2 * time.Second)
			if s.ctx.Err() != nil {
				return
			}
			if err := s.startProcess(); err != nil {
				log.Printf("failed to restart llama-server: %v", err)
				return
			}
			return
		}
	}()

	return s.waitReady(30 * time.Second)
}

func (s *Server) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}

	for time.Now().Before(deadline) {
		resp, err := client.Get(s.baseURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				log.Printf("llama-server ready")
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("llama-server not ready after %s", timeout)
}

func (s *Server) BaseURL() string {
	return s.baseURL
}

func (s *Server) Stop() {
	if s.cmd != nil && s.cmd.Process != nil {
		log.Printf("stopping llama-server (pid %d)", s.cmd.Process.Pid)
		s.cmd.Process.Signal(os.Interrupt)
		done := make(chan error, 1)
		go func() { done <- s.cmd.Wait() }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			s.cmd.Process.Kill()
		}
	}
}
