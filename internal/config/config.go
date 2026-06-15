package config

import (
	"os"
	"strconv"
)

type Config struct {
	Port           string
	ModelPath      string
	LlamaServerBin string
	LlamaPort      string
	Threads        int
	CtxSize        int
	MaxTokens      int
	NumSuggestions int
	Temperature    float64
	TopK           int
	TopP           float64
	APIKey         string
	PromptPrefix   string
}

const defaultPromptPrefix = ``

func Load() *Config {
	return &Config{
		Port:           envOr("PORT", "8400"),
		ModelPath:      envOr("MODEL_PATH", "/data/smollm2-135m-q8.gguf"),
		LlamaServerBin: envOr("LLAMA_SERVER_BIN", "/usr/local/bin/llama-server"),
		LlamaPort:      envOr("LLAMA_PORT", "8401"),
		Threads:        envInt("THREADS", 4),
		CtxSize:        envInt("CTX_SIZE", 1024),
		MaxTokens:      envInt("MAX_TOKENS", 10),
		NumSuggestions: envInt("NUM_SUGGESTIONS", 5),
		Temperature:    envFloat("TEMPERATURE", 0.3),
		TopK:           envInt("TOP_K", 20),
		TopP:           envFloat("TOP_P", 0.85),
		APIKey:         os.Getenv("API_KEY"),
		PromptPrefix:   envOr("PROMPT_PREFIX", defaultPromptPrefix),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}
