# ──────────────────────────────────────────────────────────────────────────
# Llama 4 MoE Prompt Router & Debugger — backend image
#
# Two build profiles via the INSTALL_ML build-arg:
#   * INSTALL_ML=true  (default) → full stack incl. torch + transformers,
#                                   ready to load real Llama 4 weights.
#   * INSTALL_ML=false           → lightweight image (no torch). Serves the
#                                   MOCK backend and the dashboard, and runs the
#                                   weight-free test-suite. Used by CI.
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ARG INSTALL_ML=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# System deps kept minimal; build-essential only when compiling ML wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install either the full ML stack or just the serving/runtime subset.
RUN python -m pip install --upgrade pip && \
    if [ "$INSTALL_ML" = "true" ]; then \
        pip install -r requirements.txt ; \
    else \
        pip install \
            fastapi "uvicorn[standard]" pydantic numpy httpx sse-starlette \
            streamlit pandas plotly requests pytest ; \
    fi

COPY . .

# Captured-trace / cache dir; mount a volume here in production.
RUN mkdir -p /cache/huggingface

EXPOSE 8000 8501

# Default: run the FastAPI backend. Override CMD to launch the dashboard, e.g.
#   docker run ... llama4-moe-debugger \
#     streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
