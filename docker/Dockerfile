FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra search --extra rag --extra platform

COPY app ./app
RUN uv sync --frozen --no-editable --extra search --extra rag --extra platform

FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN useradd --create-home --uid 10001 globex
COPY --from=builder --chown=globex:globex /app/.venv /app/.venv
COPY --chown=globex:globex app ./app
COPY --chown=globex:globex knowledge ./knowledge

USER globex
EXPOSE 8000
CMD ["uvicorn", "app.presentation.server:app", "--host", "0.0.0.0", "--port", "8000"]
