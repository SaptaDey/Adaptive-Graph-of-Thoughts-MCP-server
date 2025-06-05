# Stage 1: Build stage - Install dependencies using Poetry with minimal base
FROM python:3.13.3-slim-bookworm@sha256:914bf5c12ea40a97a78b2bff97fbdb766cc36ec903bfb4358faf2b74d73b555b AS builder

WORKDIR /opt/poetry

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==${POETRY_VERSION}"

# Copy lock and project files for dependency install
COPY poetry.lock pyproject.toml ./

# Configure and install dependencies without dev tools
RUN poetry config virtualenvs.create false && \
    poetry lock --no-update && \
    poetry install --only=main --no-interaction --no-ansi && \
    poetry cache clear pypi --all && \
    pip uninstall -y poetry

# Stage 2: Runtime stage - Minimal final image
FROM python:3.13.3-slim-bookworm@sha256:914bf5c12ea40a97a78b2bff97fbdb766cc36ec903bfb4358faf2b74d73b555b AS runtime

# Install only essential runtime dependencies including Node.js for Inspector
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 \
    wget \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Environment variables for Python and MCP transport default
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    APP_HOME=/app \
    PYTHONHASHSEED=random \
    MCP_TRANSPORT_TYPE=http

WORKDIR ${APP_HOME}

# Create a non-root user
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Copy dependencies and binaries from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and configuration
COPY ./src ./src
COPY ./config ./config

# Copy and enable entrypoint script
COPY ./scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set file ownership to non-root user
RUN chown -R appuser:appuser ${APP_HOME} ./config

# Switch to non-root user
USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Add a health check for the application
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint for flexible transport startup
ENTRYPOINT ["/entrypoint.sh"]
CMD ["http"]
