# ═══════════════════════════════════════════════════════════════
# Anti-Gravity AML System — Dockerfile
# Build:  docker build -t antigravity-aml .
# Run:    docker run -p 8501:8501 antigravity-aml
# ═══════════════════════════════════════════════════════════════

# Stage 1: Builder (install heavy deps)
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# System lib for networkx/scipy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy project source
COPY . .

# Create data directories
RUN mkdir -p data/processed data/models data/shap data/graphs

# Streamlit configuration
RUN mkdir -p ~/.streamlit && cat > ~/.streamlit/config.toml << 'EOF'
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#3b82f6"
backgroundColor = "#0d0f14"
secondaryBackgroundColor = "#111827"
textColor = "#e2e8f0"

[browser]
gatherUsageStats = false
EOF

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entry point
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501"]
