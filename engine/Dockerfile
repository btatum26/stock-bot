FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# Add a build argument that defaults to --no-dev
ARG UV_INSTALL_ARGS="--no-dev"

COPY pyproject.toml uv.lock ./
# Use the argument here
RUN uv sync --frozen $UV_INSTALL_ARGS

COPY . .