# ── Base image ──────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dependencies ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ────────────────────────────────────────
COPY . .

# ── Metadata labels ─────────────────────────────────────────
LABEL maintainer="OpenEnv Team" \
      version="1.0.0" \
      description="OpenEnv Customer Support Ticket Resolution Environment" \
      org.opencontainers.image.source="https://github.com/openenv/openenv-support"

# ── Expose port for HF Space ────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# ── Default command: serve HTTP API for HF Space ─────────────
CMD ["python", "server.py"]
