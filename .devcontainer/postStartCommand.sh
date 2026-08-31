#!/usr/bin/env bash


# install project with all dependencies (no GPU extras in devcontainer)
uv sync --all-groups

# install pre-commit hooks
uv run pre-commit install --install-hooks
