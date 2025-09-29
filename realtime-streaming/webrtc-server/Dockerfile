# Multi-stage build for optimized WebRTC server
FROM rust:1.74-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    librdkafka-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifest
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src/ src/

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -m -d /app webrtc

# Copy binary from builder
COPY --from=builder /app/target/release/webrtc-server /usr/local/bin/webrtc-server

# Set ownership
RUN chown webrtc:webrtc /usr/local/bin/webrtc-server

# Switch to non-root user
USER webrtc
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Expose ports
EXPOSE 8090

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/webrtc-server"]
CMD ["--bind", "0.0.0.0:8090"]