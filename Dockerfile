# Multi-stage Docker build for PINN deployment
# Build stage
FROM rust:1.75-slim as builder

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build the application with PINN features
RUN cargo build --release --features pinn

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash pinn
USER pinn

# Set working directory
WORKDIR /app

# Copy the compiled binary
COPY --from=builder /app/target/release/kwavers /app/kwavers

# Expose port for API server
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./kwavers"]
