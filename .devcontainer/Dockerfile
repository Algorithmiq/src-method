FROM ghcr.io/astral-sh/uv:latest AS uv

FROM mcr.microsoft.com/devcontainers/base:1.2.5-noble

COPY --from=uv --chown=vscode: /uv /uvx /bin/
