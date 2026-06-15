package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type CompletionRequest struct {
	Prompt        string   `json:"prompt"`
	NPredict      int      `json:"n_predict"`
	Temperature   float64  `json:"temperature"`
	TopK          int      `json:"top_k"`
	TopP          float64  `json:"top_p"`
	RepeatPenalty float64  `json:"repeat_penalty,omitempty"`
	Stop          []string `json:"stop,omitempty"`
	Stream        bool     `json:"stream"`
}

type CompletionResponse struct {
	Content string `json:"content"`
	Timings struct {
		PromptMS           float64 `json:"prompt_ms"`
		PredictedMS        float64 `json:"predicted_ms"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
	} `json:"timings"`
}

type Client struct {
	baseURL    string
	httpClient *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (c *Client) Complete(req CompletionRequest) (*CompletionResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(c.baseURL+"/completion", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("completion request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("completion failed (%d): %s", resp.StatusCode, string(data))
	}

	var result CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}

func (c *Client) Health() error {
	resp, err := c.httpClient.Get(c.baseURL + "/health")
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unhealthy: %d", resp.StatusCode)
	}
	return nil
}
