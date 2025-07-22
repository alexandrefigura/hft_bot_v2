# Multi-stage build for minimal image size
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     g++     make     && rm -rf /var/lib/apt/lists/*

# Create wheels
WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel &&     pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Final stage
FROM python:3.11-slim

# Security: run as non-root
RUN useradd -m -u 1000 botuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends     libgomp1     curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir /wheels/* &&     rm -rf /wheels

# Copy application
COPY --chown=botuser:botuser . .
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3     CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["hft", "run"]
