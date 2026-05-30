# ============================================================================
# Octopus BFT — Multi-stage Docker Build
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

# Build the octopus binary with static linking for Alpine
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -extldflags '-static'" \
    -o /octopus \
    ./cmd/octopus

# Also build the simulation/DAS tools if present (create empty placeholder if build fails)
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /octopus-sim \
    ./cmd/octopus-sim 2>/dev/null || touch /octopus-sim

# Also build genesis tool for in-container manifest generation
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /octopus-genesis \
    ./cmd/octopus-genesis 2>/dev/null || touch /octopus-genesis

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM alpine:3.19

RUN apk add --no-cache ca-certificates curl jq

# Create non-root user for security
RUN addgroup -S octopus && adduser -S octopus -G octopus

COPY --from=builder /octopus /usr/local/bin/octopus
RUN chmod +x /usr/local/bin/octopus

# Copy optional tools (placeholders created by builder if build fails)
COPY --from=builder /octopus-sim /usr/local/bin/octopus-sim
COPY --from=builder /octopus-genesis /usr/local/bin/octopus-genesis
RUN chmod +x /usr/local/bin/octopus-sim /usr/local/bin/octopus-genesis 2>/dev/null || true

# Data directory for persistent state
RUN mkdir -p /data && chown octopus:octopus /data
VOLUME /data

# P2P port (libp2p)
EXPOSE 8080
# HTTP admin/metrics port
EXPOSE 9000

USER octopus

# Health check via HTTP metrics endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:9000/metrics || exit 1

ENTRYPOINT ["octopus"]
