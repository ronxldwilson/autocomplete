## Stage 1: Download llama.cpp binaries and model
FROM debian:bookworm-slim AS assets

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ARG LLAMA_VERSION=b9637
RUN mkdir -p /out/llama /out/data && \
    curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz" \
    -o /tmp/llama.tar.gz && \
    tar xzf /tmp/llama.tar.gz -C /tmp && \
    cp /tmp/llama-${LLAMA_VERSION}/* /out/llama/ && \
    chmod +x /out/llama/llama-server && \
    rm -rf /tmp/llama*

# Use local fine-tuned model if present, otherwise download base model
COPY model.ggu[f] /out/data/
RUN if [ ! -f /out/data/model.gguf ]; then \
      curl -sL "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf" \
      -o /out/data/model.gguf; \
    fi

## Stage 2: Build Go binary
FROM golang:1.24-bookworm AS builder
WORKDIR /build
COPY go.mod ./
COPY go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /server ./cmd/server

## Stage 3: Runtime
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Keep llama binaries and libs together in one dir (backend discovery requires this)
COPY --from=assets /out/llama/ /opt/llama/
COPY --from=assets /out/data/model.gguf /data/model.gguf
COPY --from=builder /server /server

ENV LD_LIBRARY_PATH=/opt/llama
ENV MODEL_PATH=/data/model.gguf
ENV LLAMA_SERVER_BIN=/opt/llama/llama-server
ENV PORT=8400

EXPOSE 8400

CMD ["/server"]
