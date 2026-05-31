# ============================================================================
# Evolv-BFT — Multi-stage Docker Build
# ============================================================================
# Stage 1: Build the Go binary
# Stage 2: Minimal runtime image
# ============================================================================

# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git gcc musl-dev

WORKDIR /build

# Copy go module files first for layer caching
COPY src/go.mod src/go.sum ./
RUN go mod download

# Copy source code
COPY src/ .

# Build the evolvbft binary with static linking for Alpine
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -extldflags '-static'" \
    -o /evolvbft \
    ./cmd/evolvbft

# Also build the simulation/DAS tools if present (create empty placeholder if build fails)
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /evolvbft-sim \
    ./cmd/evolvbft-sim 2>/dev/null || touch /evolvbft-sim

# Also build genesis tool for in-container manifest generation
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /evolvbft-genesis \
    ./cmd/evolvbft-genesis 2>/dev/null || touch /evolvbft-genesis

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM alpine:3.19

RUN apk add --no-cache ca-certificates curl jq

# Create non-root user for security
RUN addgroup -S evolvbft && adduser -S evolvbft -G evolvbft

COPY --from=builder /evolvbft /usr/local/bin/evolvbft
RUN chmod +x /usr/local/bin/evolvbft

# Copy optional tools (placeholders created by builder if build fails)
COPY --from=builder /evolvbft-sim /usr/local/bin/evolvbft-sim
COPY --from=builder /evolvbft-genesis /usr/local/bin/evolvbft-genesis
RUN chmod +x /usr/local/bin/evolvbft-sim /usr/local/bin/evolvbft-genesis 2>/dev/null || true

# Data directory for persistent state
RUN mkdir -p /data && chown evolvbft:evolvbft /data
VOLUME /data

# P2P port (libp2p)
EXPOSE 8080
# HTTP admin/metrics port
EXPOSE 9000

USER evolvbft

# Health check via HTTP metrics endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:9000/metrics || exit 1

ENTRYPOINT ["evolvbft"]
