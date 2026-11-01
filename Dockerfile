
# Use a lightweight Python image
FROM python:3.12-slim

# Copy uv from the official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Configure uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

# Copy dependency files first for better Docker caching
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY app/ ./app/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uv", "run", "--no-sync", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
