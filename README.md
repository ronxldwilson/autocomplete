# Autocomplete

LLM-powered search query autocomplete service for B2B procurement. Uses a fine-tuned Qwen2.5-0.5B model served via llama.cpp, wrapped in a Go HTTP server.

## Architecture

```
Client → Go HTTP Server (:8400) → llama-server (:8401) → GGUF Model
```

- **Go server** manages llama-server as a subprocess with auto-restart on crash
- **llama.cpp** handles model inference with parallel completion slots
- **Fine-tuned model** trained on 22k B2B procurement queries via LoRA

## Quick Start

```bash
# Build and run with Docker
docker compose up -d

# Test
curl -X POST http://localhost:8400/autocomplete \
  -H "Content-Type: application/json" \
  -d '{"text": "dental chair"}'
```

The model GGUF file must be placed at the project root as `model.gguf` before building. See [Fine-tuning](#fine-tuning) to create one.

## API

### POST /autocomplete

```json
{
  "text": "stainless steel",
  "max_tokens": 4,
  "num_suggestions": 5
}
```

Response:
```json
{
  "suggestions": [
    {
      "completion": "pipes for industrial use",
      "full_text": "stainless steel pipes for industrial use",
      "latency_ms": 150
    }
  ]
}
```

### GET /health

Returns `{"status": "ok"}` when llama-server is ready.

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8400 | HTTP server port |
| `THREADS` | 4 | CPU threads for inference |
| `CTX_SIZE` | 64 | Context window size |
| `MAX_TOKENS` | 4 | Max tokens to generate |
| `NUM_SUGGESTIONS` | 5 | Suggestions per request |
| `TEMPERATURE` | 0.3 | Sampling temperature |
| `TOP_K` | 20 | Top-K sampling |
| `TOP_P` | 0.85 | Top-P (nucleus) sampling |
| `API_KEY` | | Optional API key for auth |

## Fine-tuning

### Data Preparation

```bash
# Requires a directory of query JSON files
python finetune/prepare_data.py
```

Generates `train.jsonl` and `val.jsonl` with prefix-completion pairs.

### Training

**Local (CPU/MPS):**
```bash
cd autocomplete
uv run python finetune/train.py
```

**Google Colab (GPU):**
Upload `finetune/finetune_qwen.ipynb` to Colab with `train.jsonl` and `val.jsonl`.

### GGUF Conversion

```bash
uv run python finetune/convert_to_gguf.py
```

Converts the merged HuggingFace model to GGUF Q8_0 format. Further quantize with llama-quantize for smaller/faster variants (Q4_K_M, Q3_K_S).

## Benchmarks

```bash
# Set the service URL
export AUTOCOMPLETE_URL=http://localhost:8400/autocomplete

# Quick 20-query test
python finetune/benchmark.py

# Full 100-query test (mixed full/partial/long queries)
python finetune/benchmark_100.py

# Compare quantization levels
python finetune/benchmark_quant.py
```

### Results (Qwen2.5-0.5B Q3_K_S, 4-core CPU)

| Metric | Value |
|--------|-------|
| Query hit rate | 83-91% |
| Avg latency | 244ms |
| Suggestions/query | 4.1 |
| Hinglish contamination | <1% |

## Docker

```bash
# Build (model.gguf must be in project root)
docker build -t autocomplete:latest .

# Run
docker compose up -d
```

Memory usage depends on model size:
- SmolLM2-135M Q8: ~768MB
- Qwen2.5-0.5B Q4: ~1GB
- Qwen2.5-0.5B Q3: ~900MB
